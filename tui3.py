from __future__ import annotations

"""agent_tui.py — dense Rich TUI for the Agent runtime.

Layout:
- large chat-like event timeline
- right-side status + tool context panel
- small bottom panel for the latest thought / action
- animated spinner during thinking / tool execution / validation
"""

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import questionary
from questionary import Style as QStyle

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from src.agent import Agent, ModelSource


# ── palette ────────────────────────────────────────────────────────────────

C = {
    "accent": "#00e5a0",
    "purple": "#7c6af7",
    "warn": "#f5a623",
    "err": "#f05252",
    "muted": "#555560",
    "text": "#d4d4db",
    "dim": "#3a3a45",
    "think": "#a78bfa",
    "action": "#fbbf24",
    "valid": "#7ab4f5",
    "done": "#34d399",
    "info": "#94a3b8",
}

BADGE_STYLES = {
    "AGENT_STARTED": ("agent", C["accent"], "◈ STARTED"),
    "INTENT_EXTRACTED": ("agent", C["accent"], "◈ INTENT"),
    "STEP_STARTED": ("info", C["info"], "· STEP"),
    "CONTEXT_BUILT": ("info", C["info"], "· CONTEXT"),
    "THINKING_STARTED": ("think", C["think"], "⟳ THINKING"),
    "THINKING_ENDED": ("think", C["think"], "⟳ THOUGHT"),
    "ACTION_PLANNED": ("action", C["action"], "▶ PLAN"),
    "ACTION_EXECUTING": ("action", C["action"], "▶ EXEC"),
    "ACTION_EXECUTED": ("action", C["action"], "▶ DONE"),
    "VALIDATION_STARTED": ("valid", C["valid"], "✦ VALIDATE"),
    "VALIDATION_DONE": ("valid", C["valid"], "✦ RESULT"),
    "MEMORY_UPDATED": ("info", C["info"], "· MEMORY"),
    "AGENT_DONE": ("done", C["done"], "✓ DONE"),
    "AGENT_ERROR": ("err", C["err"], "✗ ERROR"),
    "TOOL_ERROR": ("err", C["err"], "✗ TOOL ERR"),
    "WARNING": ("warn", C["warn"], "⚠ WARN"),
}

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# ── helpers ────────────────────────────────────────────────────────────────


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _short(value: Any, limit: int = 120) -> str:
    try:
        if isinstance(value, str):
            s = value
        else:
            s = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        s = str(value)
    s = s.replace("", " ").replace("", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _kv_lines(data: dict[str, Any], indent: int = 0, max_items: int = 16) -> list[str]:
    lines: list[str] = []
    prefix = "  " * indent
    items = list(data.items())[:max_items]
    for k, v in items:
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.extend(_kv_lines(v, indent + 1))
        elif isinstance(v, list):
            if not v:
                lines.append(f"{prefix}{k}: []")
            elif all(not isinstance(x, (dict, list)) for x in v[:5]):
                lines.append(f"{prefix}{k}: [{', '.join(_short(x, 40) for x in v[:5])}{' …' if len(v) > 5 else ''}]")
            else:
                lines.append(f"{prefix}{k}:")
                for idx, item in enumerate(v[:5]):
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  - [{idx}]")
                        lines.extend(_kv_lines(item, indent + 2))
                    else:
                        lines.append(f"{prefix}  - {_short(item, 80)}")
                if len(v) > 5:
                    lines.append(f"{prefix}  … +{len(v) - 5} more")
        else:
            lines.append(f"{prefix}{k}: {_short(v, 120)}")
    if len(data) > max_items:
        lines.append(f"{prefix}… +{len(data) - max_items} more")
    return lines


@dataclass(slots=True)
class EventRow:
    idx: int
    ts: float
    event_type: str
    step: Optional[int]
    run_id: str
    payload: dict[str, Any]
    duration_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def time_label(self) -> str:
        return datetime.fromtimestamp(self.ts).strftime("%H:%M:%S")


# ── event renderer ─────────────────────────────────────────────────────────


class AgentTUI:
    """Rich Live TUI with a compact timeline, status panel, and latest-thought footer."""

    def __init__(self, max_events: int = 100):
        self.console = Console()
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._think_start: Optional[float] = None
        self._run_id: Optional[str] = None
        self._current_step = 0
        self._event_count = 0
        self._events: deque[EventRow] = deque(maxlen=max_events)
        self._selected_idx: int = -1
        self._available_tools: list[Any] = []
        self._current_action: dict[str, Any] = {}
        self._current_tool: str = ""
        self._current_reason: str = ""
        self._latest_validation: dict[str, Any] = {}
        self._spinner_label: str = ""
        self._spinner_color: str = C["accent"]
        self._spinner_index: int = 0
        self._live: Optional[Live] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._stop_spin = threading.Event()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=12,
            transient=False,
            screen=False,
        )
        self._live.start()
        self._stop_spin.clear()
        if self._spin_thread is None or not self._spin_thread.is_alive():
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()

    def on_event(self, event) -> None:
        with self._lock:
            self._push_event(event)
            etype = event.type.name if hasattr(event.type, "name") else str(event.type)

            if etype == "THINKING_STARTED":
                self._set_spinner("thinking", C["think"])
            elif etype == "THINKING_ENDED":
                self._set_spinner("planning next step", C["think"])
            elif etype == "ACTION_EXECUTING":
                self._set_spinner(f"executing {self._current_tool or 'tool'}", C["action"])
            elif etype == "ACTION_EXECUTED":
                self._set_spinner("validating result", C["valid"])
            elif etype == "VALIDATION_STARTED":
                self._set_spinner("validating", C["valid"])
            elif etype in ("AGENT_DONE", "AGENT_ERROR", "TOOL_ERROR"):
                self._stop_spinner_state()

            self._refresh()

    def finish(self) -> None:
        with self._lock:
            self._stop_spinner_state()
            self._refresh(final=True)
            if self._live is not None:
                try:
                    self._live.stop()
                finally:
                    self._live = None
            self._stop_spin.set()

    # ── spinner driver ─────────────────────────────────────────────────────

    def _spin_loop(self) -> None:
        while not self._stop_spin.is_set():
            with self._lock:
                if self._live is not None and self._spinner_label:
                    self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
                    self._refresh()
            time.sleep(0.08)

    def _set_spinner(self, label: str, color: str) -> None:
        self._spinner_label = label
        self._spinner_color = color
        self._spinner_index = 0

    def _stop_spinner_state(self) -> None:
        self._spinner_label = ""

    # ── event ingestion ───────────────────────────────────────────────────

    def _push_event(self, event) -> None:
        etype = event.type.name if hasattr(event.type, "name") else str(event.type)
        payload = event.payload or {}
        self._event_count += 1
        self._events.append(
            EventRow(
                idx=self._event_count,
                ts=getattr(event, "ts", time.time()),
                event_type=etype,
                step=getattr(event, "step", None),
                run_id=getattr(event, "run_id", self._run_id or ""),
                payload=payload,
                duration_ms=getattr(event, "duration_ms", None),
                error=getattr(event, "error", None),
            )
        )
        self._selected_idx = len(self._events) - 1

        if etype == "AGENT_STARTED":
            self._start_time = time.time()
            self._run_id = getattr(event, "run_id", None)
            self._current_action = {}
            self._current_tool = ""
            self._current_reason = ""
            self._latest_validation = {}
            self._available_tools = []
        elif etype == "STEP_STARTED":
            self._current_step = getattr(event, "step", 0) or 0
        elif etype == "CONTEXT_BUILT":
            tools = payload.get("tools") or []
            self._available_tools = list(tools) if isinstance(tools, list) else [tools]
        elif etype == "THINKING_STARTED":
            self._think_start = time.time()
        elif etype in ("THINKING_ENDED", "ACTION_PLANNED", "ACTION_EXECUTING"):
            action = payload.get("action") or {}
            if isinstance(action, dict):
                self._current_action = action
                self._current_tool = str(action.get("tool", ""))
                self._current_reason = str(action.get("reason", "")).strip()
        elif etype == "VALIDATION_DONE":
            self._latest_validation = dict(payload)

    def _refresh(self, final: bool = False) -> None:
        if self._live is None:
            return
        self._live.update(self.render(final=final), refresh=True)

    # ── formatting ────────────────────────────────────────────────────────

    def _badge(self, event_type: str) -> Text:
        _, color, label = BADGE_STYLES.get(event_type, ("info", C["info"], f"· {event_type[:10]}"))
        t = Text()
        t.append(f" {label:<13}", style=Style(color=color, bold=True))
        return t

    def _badge_color(self, event_type: str) -> str:
        return BADGE_STYLES.get(event_type, ("info", C["info"], ""))[1]

    def _elapsed(self) -> str:
        if self._start_time is None:
            return "—"
        return f"{time.time() - self._start_time:.1f}s"

    def _dur(self, ms: Optional[float]) -> Text:
        if ms is None:
            return Text("—", style=C["muted"])
        if ms < 1000:
            return Text(f"{ms:.0f}ms", style=C["muted"])
        return Text(f"{ms / 1000:.2f}s", style=C["warn"])

    def _tool_name(self, tool: Any) -> str:
        if isinstance(tool, dict):
            for key in ("name", "tool", "id"):
                value = tool.get(key)
                if value:
                    return str(value)
            return _short(tool, 40)
        for attr in ("name", "tool_name"):
            value = getattr(tool, attr, None)
            if value:
                return str(value)
        return _short(tool, 40)

    def _tool_desc(self, tool: Any) -> str:
        if isinstance(tool, dict):
            for key in ("description", "desc", "summary"):
                value = tool.get(key)
                if value:
                    return _short(value, 52)
            return ""
        for attr in ("description", "desc", "summary"):
            value = getattr(tool, attr, None)
            if value:
                return _short(value, 52)
        return ""

    def _spinner_render(self) -> Text:
        if not self._spinner_label:
            return Text("idle", style=C["muted"])
        frame = SPINNER_FRAMES[self._spinner_index]
        t = Text()
        t.append(f"{frame} ", style=Style(color=self._spinner_color, bold=True))
        t.append(self._spinner_label, style=Style(color=self._spinner_color, bold=True))
        return t

    def _event_summary(self, row: EventRow) -> Text:
        p = row.payload
        t = Text()

        if row.event_type == "AGENT_STARTED":
            t.append("query: ", style=C["muted"])
            t.append(_short(p.get("query", ""), 180), style=C["text"])
        elif row.event_type == "INTENT_EXTRACTED":
            t.append("intent: ", style=C["muted"])
            t.append(_short(p.get("intent", "—"), 120), style=C["accent"])
            goals = p.get("goals", [])
            if goals:
                t.append("  goals: ", style=C["muted"])
                t.append(_short(goals, 120), style=C["text"])
        elif row.event_type == "STEP_STARTED":
            state = p.get("state", {})
            t.append("step ", style=f"bold {C['purple']}")
            t.append(str((row.step or 0) + 1), style=f"bold {C['purple']}")
            if isinstance(state, dict):
                status = state.get("status", "")
                if status:
                    t.append("  status: ", style=C["muted"])
                    t.append(str(status), style=C["text"])
        elif row.event_type == "CONTEXT_BUILT":
            tools = p.get("tools") or []
            arts = p.get("artifacts") or []
            t.append("tools: ", style=C["muted"])
            t.append(str(len(tools) if isinstance(tools, list) else tools), style=C["text"])
            t.append("  artifacts: ", style=C["muted"])
            t.append(str(len(arts) if isinstance(arts, list) else arts), style=C["text"])
            t.append("  ctx: ", style=C["muted"])
            t.append(f"{p.get('context_len', 0)} chars", style=C["text"])
        elif row.event_type == "THINKING_STARTED":
            t.append("model reasoning started", style=C["think"])
        elif row.event_type == "THINKING_ENDED":
            action = p.get("action") or {}
            t.append("plan: ", style=C["muted"])
            t.append(_short(action, 180), style=C["purple"])
        elif row.event_type == "ACTION_PLANNED":
            action = p.get("action") or {}
            t.append("call: ", style=C["muted"])
            t.append(_short(action, 180), style=C["action"])
        elif row.event_type == "ACTION_EXECUTING":
            action = p.get("action") or {}
            if isinstance(action, dict):
                t.append("exec: ", style=C["muted"])
                t.append(str(action.get("tool", "—")), style=C["action"])
                if action.get("input"):
                    t.append("  input: ", style=C["muted"])
                    t.append(_short(action.get("input"), 110), style=C["text"])
        elif row.event_type == "ACTION_EXECUTED":
            t.append("success: ", style=C["muted"])
            ok = bool(p.get("success", True))
            t.append(str(ok).lower(), style=C["done"] if ok else C["err"])
            t.append("  summary: ", style=C["muted"])
            t.append(_short(p.get("summary", ""), 150), style=C["text"] if ok else C["err"])
        elif row.event_type == "VALIDATION_STARTED":
            t.append("validating latest step", style=C["valid"])
        elif row.event_type == "VALIDATION_DONE":
            t.append("status: ", style=C["muted"])
            t.append(str(p.get("status", "—")), style=C["valid"])
            reason = str(p.get("reason", "")).strip()
            if reason:
                t.append("  reason: ", style=C["muted"])
                t.append(reason, style=C["text"])
            missing = p.get("missing", [])
            if missing:
                t.append("  missing: ", style=C["muted"])
                t.append(_short(missing, 100), style=C["warn"])
        elif row.event_type == "MEMORY_UPDATED":
            t.append(_short(p, 160), style=C["info"])
        elif row.event_type == "AGENT_DONE":
            t.append("completed", style=f"bold {C['done']}")
        elif row.event_type in ("AGENT_ERROR", "TOOL_ERROR"):
            t.append(_short(row.error or p.get("error", "unknown error"), 170), style=f"bold {C['err']}")
        elif row.event_type == "WARNING":
            t.append("listener warning", style=C["warn"])
        else:
            t.append(_short(p, 160), style=C["muted"])

        if row.duration_ms is not None:
            t.append("  dur: ", style=C["muted"])
            t.append_text(self._dur(row.duration_ms))

        return t

    def _status_panel(self) -> Panel:
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(width=10)
        grid.add_column(ratio=1)

        grid.add_row(Text("activity", style=C["muted"]), self._spinner_render())
        grid.add_row(Text("step", style=C["muted"]), Text(str(self._current_step + 1), style=f"bold {C['purple']}"))
        grid.add_row(Text("elapsed", style=C["muted"]), Text(self._elapsed(), style=C["text"]))
        grid.add_row(Text("current", style=C["muted"]), Text(self._current_tool or "—", style=f"bold {C['action']}" if self._current_tool else C["text"]))
        grid.add_row(Text("reason", style=C["muted"]), Text(_short(self._current_reason or "—", 60), style=C["text"]))

        if self._latest_validation:
            grid.add_row(Text("val", style=C["muted"]), Text(str(self._latest_validation.get("status", "—")), style=f"bold {C['valid']}"))
            reason = str(self._latest_validation.get("reason", "")).strip()
            if reason:
                grid.add_row(Text("val reason", style=C["muted"]), Text(_short(reason, 60), style=C["text"]))

        return Panel(grid, title="live status", title_align="left", border_style=C["dim"], box=box.ROUNDED, padding=(0, 0))

    def _tool_panel(self) -> Panel:
        tools = self._available_tools or []

        header = Table.grid(expand=True, padding=(0, 1))
        header.add_column(width=10)
        header.add_column(ratio=1)
        header.add_row(Text("tools", style=C["muted"]), Text(str(len(tools)), style=C["text"]))

        list_table = Table(expand=True, box=None, show_header=False, pad_edge=False, collapse_padding=True)
        list_table.add_column("tool")
        if tools:
            for tool in tools[:16]:
                name = self._tool_name(tool)
                desc = self._tool_desc(tool)
                selected = name == self._current_tool
                row = Text()
                row.append("▶ " if selected else "  ", style=C["accent"] if selected else C["muted"])
                row.append(name, style=f"bold {C['accent']}" if selected else C["text"])
                if desc:
                    row.append(f"  {desc}", style=C["muted"])
                list_table.add_row(row)
            if len(tools) > 16:
                list_table.add_row(Text(f"… +{len(tools) - 16} more", style=C["muted"]))
        else:
            list_table.add_row(Text("no tools in context", style=C["muted"]))

        return Panel(Group(header, Text(""), list_table), title="available tools", title_align="left", border_style=C["dim"], box=box.ROUNDED, padding=(0, 1))

    def _detail_panel(self, row: EventRow) -> Panel:
        p = row.payload

        body = Group(
            Text.assemble(
                ("event: ", C["muted"]), (row.event_type, f"bold {self._badge_color(row.event_type)}"),
                ("  step: ", C["muted"]), (str(row.step if row.step is not None else "—"), C["text"]),
                ("  at: ", C["muted"]), (row.time_label, C["text"]),
            ),
            Text(""),
            Text("latest thought", style=f"bold {C['text']}"),
            self._event_summary(row),
        )

        if row.event_type in ("AGENT_ERROR", "TOOL_ERROR") and p.get("traceback"):
            body = Group(
                body,
                Text(""),
                Text("traceback", style=f"bold {C['err']}"),
                Text(str(p.get("traceback"))[-1200:], style=C["err"]),
            )

        if row.event_type in ("ACTION_EXECUTING", "ACTION_PLANNED", "THINKING_ENDED", "VALIDATION_DONE"):
            body = Group(
                body,
                Text(""),
                Text("payload", style=f"bold {C['text']}"),
                Text("".join(_kv_lines(p)) if p else "{}", style=C["text"]),
            )

        return Panel(body, title=f"detail · {row.event_type}", title_align="left", border_style=self._badge_color(row.event_type), box=box.ROUNDED, padding=(1, 1))

    def _timeline(self) -> Panel:
        table = Table(
            expand=True,
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style=f"bold {C['muted']}",
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("#", width=4, no_wrap=True)
        table.add_column("time", width=8, no_wrap=True)
        table.add_column("event", width=16, no_wrap=True)
        table.add_column("summary", ratio=1)
        table.add_column("dur", width=8, no_wrap=True)

        for row in list(self._events)[-20:]:
            color = self._badge_color(row.event_type)
            summary = self._event_summary(row)
            selected = row.idx == self._event_count
            idx_style = f"bold {color}" if selected else C["muted"]
            event_style = f"bold {color}" if selected else color
            dur = self._dur(row.duration_ms) if row.duration_ms is not None else Text("—", style=C["muted"])
            table.add_row(
                Text(str(row.idx), style=idx_style),
                Text(row.time_label, style=C["muted"]),
                Text(row.event_type, style=event_style),
                summary,
                dur,
            )

        return Panel(table, title="recent events", title_align="left", border_style=C["dim"], box=box.ROUNDED)

    def _header(self) -> Panel:
        title = Text.assemble(
            ("agent", f"bold {C['accent']}"),
            (" runtime", f"bold {C['text']}"),
            (" tui", f"bold {C['purple']}"),
        )
        subtitle = Text.assemble(("dense event view · reason · tool call · validation · errors", C["muted"]))
        content = Align.center(Text.assemble(title, "", subtitle))
        run = self._run_id[:8] if self._run_id else "—"
        return Panel.fit(content, border_style=C["dim"], padding=(1, 4), title=f"run {run}", title_align="left")

    def _footer(self) -> Panel:
        if not self._events:
            help_text = Text.assemble(
                ("latest thought will appear here once the agent starts. ", C["muted"]),
                ("The top area stays chat-like and easy to scan.", C["text"]),
            )
            return Panel(help_text, border_style=C["dim"], box=box.ROUNDED)

        row = self._events[-1]
        tool = self._current_tool or (self._current_action.get("tool") if isinstance(self._current_action, dict) else None) or "—"
        reason = self._current_reason or (self._current_action.get("reason") if isinstance(self._current_action, dict) else "") or "—"

        body = Group(
            Text.assemble(
                ("latest thought: ", C["muted"]),
                (str(tool), f"bold {C['action']}"),
                ("  step: ", C["muted"]),
                (str((self._current_step or 0) + 1), C["text"]),
                ("  activity: ", C["muted"]),
                (self._spinner_render(), ""),
            ),
            Text(_short(reason, 180), style=C["text"]),
            Text.assemble(
                ("last event: ", C["muted"]),
                (row.event_type, C["text"]),
                ("  dur: ", C["muted"]),
                (str(self._dur(row.duration_ms)), C["text"]),
            ),
        )
        return Panel(body, title="latest thought", title_align="left", border_style=C["dim"], box=box.ROUNDED, padding=(0, 1))

    # ── render tree ───────────────────────────────────────────────────────

    def render(self, final: bool = False):
        root = Layout(name="root")
        root.split_column(
            Layout(name="header", size=5),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=7),
        )
        root["body"].split_row(
            Layout(name="timeline", ratio=3),
            Layout(name="sidebar", ratio=1),
        )

        root["header"].update(self._header())
        root["timeline"].update(self._timeline())
        root["sidebar"].update(Group(self._status_panel(), Text(""), self._tool_panel()))

        if final and self._events:
            footer_panel = Panel(
                Text.assemble(
                    ("run finished · ", C["done"]),
                    (f"events={self._event_count}", C["text"]),
                    ("  elapsed=", C["muted"]),
                    (self._elapsed(), C["accent"]),
                ),
                border_style=C["dim"],
                box=box.ROUNDED,
            )
        else:
            footer_panel = self._footer()

        root["footer"].update(footer_panel)
        return root


# ── questionary theme ───────────────────────────────────────────────────────

Q_STYLE = QStyle(
    [
        ("qmark", "fg:#7c6af7 bold"),
        ("question", "fg:#d4d4db bold"),
        ("answer", "fg:#00e5a0 bold"),
        ("pointer", "fg:#7c6af7 bold"),
        ("highlighted", "fg:#00e5a0 bold"),
        ("selected", "fg:#00e5a0"),
        ("separator", "fg:#3a3a45"),
        ("instruction", "fg:#555560"),
        ("text", "fg:#d4d4db"),
        ("disabled", "fg:#555560 italic"),
    ]
)


# ── main loop ───────────────────────────────────────────────────────────────


def main() -> None:
    console = Console()

    console.print()
    console.print(
        Panel.fit(
            Align.center(
                Text.assemble(
                    ("agent", f"bold {C['accent']}"),
                    (" runtime", f"bold {C['text']}"),
                    (" tui", f"bold {C['purple']}"),
                    ("dense event timeline + status panel", C["muted"]),
                )
            ),
            border_style=C["dim"],
            padding=(1, 4),
        )
    )
    console.print()

    while True:
        action = questionary.select(
            "what would you like to do?",
            choices=[
                questionary.Choice("▶  run agent with a query", value="run"),
                questionary.Choice("⚙  configure options", value="config"),
                questionary.Choice("✕  exit", value="exit"),
            ],
            style=Q_STYLE,
        ).ask()

        if action is None or action == "exit":
            console.print(f"[{C['muted']}]goodbye.[/]")
            break

        if action == "config":
            console.print(f"[{C['muted']}]→ the live panel now shows status, tools, spinner, and latest thought.[/]")
            continue

        if action == "run":
            query = questionary.text(
                "query:",
                style=Q_STYLE,
                validate=lambda v: True if v.strip() else "please enter a query",
            ).ask()

            if not query or not query.strip():
                continue

            tui = AgentTUI(max_events=120)
            tui.start()

            agent = Agent(ModelSource("./models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"))
            agent.event_bus.subscribe(tui.on_event)

            t = threading.Thread(target=agent.run, args=(query.strip(),), daemon=True)
            t.start()
            t.join()

            tui.finish()

            again = questionary.confirm("run another query?", style=Q_STYLE, default=True).ask()
            if not again:
                console.print(f"[{C['muted']}]goodbye.[/]")
                break


if __name__ == "__main__":
    main()
