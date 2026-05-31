"""ARC TUI — Action & Reasoning Core terminal interface.

Usage:
    from tui import run
    from src.agent import Agent
    from src.llama_runtime import ModelSource

    run(agent_factory=lambda: Agent(ModelSource.from_path("model.gguf")))

Or from the CLI (shows full launch screen):
    python -m tui
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Input, RichLog, Static

from src.events import Event, EventType


# ── Cross-thread message ──────────────────────────────────────────────────────


class AgentEvent(Message):
    def __init__(self, event: Any) -> None:
        super().__init__()
        self.event = event


# ── Launch Screen ─────────────────────────────────────────────────────────────


class LaunchScreen(Screen):
    """Startup configuration screen — model config + initial query.

    Dismissed with a dict ``{"query": str, ...model params...}`` on launch,
    or ``None`` if the user quits before submitting.
    """

    CSS = """
    LaunchScreen {
        align: center middle;
        background: #0B0F14;
    }
    #launch-panel {
        width: 70;
        height: auto;
        border: tall #1E293B;
        background: #0D1219;
        padding: 2 3;
    }
    #launch-logo {
        color: #3BE8FF;
        text-style: bold;
        margin-bottom: 0;
    }
    #launch-subtitle {
        color: #1E293B;
        margin-bottom: 1;
    }
    .launch-sep {
        color: #1E293B;
        margin-bottom: 1;
    }
    .launch-section {
        color: #4D7CFE;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }
    .launch-label {
        color: #475569;
        margin-bottom: 0;
    }
    .launch-input {
        background: #0B0F14;
        border: tall #1E293B;
        color: #E2E8F0;
        margin-bottom: 1;
        height: 3;
    }
    .launch-input:focus {
        border: tall #4D7CFE;
    }
    .half {
        width: 1fr;
        margin-right: 1;
    }
    .half:last-child {
        margin-right: 0;
    }
    .half Input {
        background: #0B0F14;
        border: tall #1E293B;
        color: #E2E8F0;
        margin-bottom: 1;
        height: 3;
    }
    .half Input:focus {
        border: tall #4D7CFE;
    }
    #launch-query {
        border: tall #1E293B;
        height: 3;
    }
    #launch-query:focus {
        border: tall #3BE8FF;
    }
    #launch-btn {
        margin-top: 1;
        background: #4D7CFE;
        color: #E2E8F0;
        border: none;
        width: 100%;
        height: 3;
    }
    #launch-btn:hover {
        background: #3BE8FF;
        color: #0B0F14;
    }
    #launch-footer {
        color: #1E293B;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("ctrl+c", "quit", "Quit", show=False)]

    def __init__(self, has_factory: bool = False) -> None:
        super().__init__()
        self._has_factory = has_factory

    def compose(self) -> ComposeResult:
        with Vertical(id="launch-panel"):
            yield Static("  ARC  Action & Reasoning Core", id="launch-logo")
            yield Static("─" * 62, classes="launch-sep")

            if not self._has_factory:
                yield Static("MODEL", classes="launch-section")
                yield Static("path", classes="launch-label")
                yield Input(
                    value="./models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
                    id="model-path",
                    classes="launch-input",
                )
                with Horizontal():
                    with Vertical(classes="half"):
                        yield Static("n_ctx", classes="launch-label")
                        yield Input(value="6000", id="n-ctx")
                    with Vertical(classes="half"):
                        yield Static("gpu layers", classes="launch-label")
                        yield Input(value="0", id="n-gpu")
                    with Vertical(classes="half"):
                        yield Static("chat format", classes="launch-label")
                        yield Input(placeholder="auto", id="chat-fmt")

                yield Static("─" * 62, classes="launch-sep")

            yield Static("QUERY", classes="launch-section")
            yield Input(
                placeholder="what should the agent do?",
                id="launch-query",
                classes="launch-input",
            )

            yield Button("  Launch", id="launch-btn")
            yield Static("enter to launch  ·  ctrl+c to quit", id="launch-footer")

    def on_mount(self) -> None:
        self.query_one("#launch-query", Input).focus()

    @on(Input.Submitted, "#launch-query")
    def on_query_submitted(self) -> None:
        self._do_launch()

    @on(Button.Pressed, "#launch-btn")
    def on_launch_pressed(self) -> None:
        self._do_launch()

    def _do_launch(self) -> None:
        query = self.query_one("#launch-query", Input).value.strip()
        if not query:
            self.query_one("#launch-query", Input).focus()
            return

        config: dict[str, Any] = {"query": query}

        if not self._has_factory:
            config["model_path"] = self.query_one("#model-path", Input).value.strip()
            try:
                config["n_ctx"] = int(self.query_one("#n-ctx", Input).value or "6000")
            except ValueError:
                config["n_ctx"] = 6000
            try:
                config["n_gpu_layers"] = int(
                    self.query_one("#n-gpu", Input).value or "0"
                )
            except ValueError:
                config["n_gpu_layers"] = 0
            fmt = self.query_one("#chat-fmt", Input).value.strip() or None
            config["chat_format"] = fmt

        self.dismiss(config)


# ── Widgets ───────────────────────────────────────────────────────────────────


class ArcHeader(Widget):
    """Top status bar — logo, status badge, step counter, elapsed timer."""

    DEFAULT_CSS = """
    ArcHeader {
        dock: top;
        height: 1;
        background: #0D1219;
        color: #94A3B8;
        padding: 0 1;
    }
    """

    status: reactive[str] = reactive("idle")
    step: reactive[int] = reactive(0)
    artifact_count: reactive[int] = reactive(0)

    def __init__(self) -> None:
        super().__init__()
        self._start_time: float | None = None

    def render(self) -> Text:
        t = Text()
        t.append("  ARC  ", style="bold #3BE8FF")
        t.append("│ ", style="#1E293B")

        status_map = {
            "running": ("●", "bold #3BE8FF", "RUNNING  "),
            "paused": ("●", "bold #F59E0B", "PAUSED   "),
            "done": ("●", "bold #22C55E", "DONE     "),
            "error": ("●", "bold #EF4444", "ERROR    "),
            "idle": ("○", "#1E293B", "IDLE     "),
        }
        dot, dot_style, label = status_map.get(
            self.status, ("○", "#1E293B", "IDLE     ")
        )
        t.append(f"{dot} ", style=dot_style)
        t.append(label, style=dot_style)

        t.append("│ ", style="#1E293B")
        t.append("step ", style="#94A3B8")
        t.append(str(self.step), style="#E2E8F0")

        if self._start_time is not None:
            elapsed = int(time.time() - self._start_time)
            m, s = divmod(elapsed, 60)
            t.append(f"  {m:02d}:{s:02d}", style="#94A3B8")

        t.append("  │ ", style="#1E293B")
        t.append("artifacts ", style="#94A3B8")
        t.append(str(self.artifact_count), style="#E2E8F0")

        t.append("  │ ", style="#1E293B")
        t.append("Action & Reasoning Core", style="#1E293B")
        return t


class StatePanel(Widget):
    """Left panel — intent, goals, constraints, live status."""

    DEFAULT_CSS = """
    StatePanel {
        height: 1fr;
        border: tall #1E293B;
        background: #0B0F14;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._state: dict[str, Any] = {}

    def update_state(self, data: dict[str, Any]) -> None:
        self._state = data
        self.refresh()

    def render(self) -> Text:
        t = Text()
        s = self._state

        t.append(" STATE\n", style="#94A3B8")
        t.append("─" * 22 + "\n", style="#1E293B")

        def row(key: str, val: str, val_style: str = "#E2E8F0") -> None:
            t.append(f" {key:<13}", style="#4D7CFE")
            t.append(f"{val}\n", style=val_style)

        intent = s.get("intent") or "—"
        row("intent", intent[:20], "#3BE8FF")

        status = s.get("status", "")
        status_style = {
            "working": "#F59E0B",
            "done": "#22C55E",
            "new": "#94A3B8",
            "planned": "#8B5CF6",
            "present": "#3BE8FF",
            "replan": "#F59E0B",
        }.get(status, "#E2E8F0")
        row("status", status or "—", status_style)
        row("step", str(s.get("step_index", 0)))

        goals = s.get("goals", [])
        if goals:
            t.append(f" {'goals':<13}", style="#4D7CFE")
            for i, g in enumerate(goals[:4]):
                prefix = " " * 14 if i > 0 else ""
                t.append(f"{prefix}{g[:18]}\n", style="#E2E8F0")

        constraints = s.get("constraints", [])
        if constraints:
            t.append(f" {'constraints':<13}", style="#4D7CFE")
            for i, c in enumerate(constraints[:3]):
                prefix = " " * 14 if i > 0 else ""
                t.append(f"{prefix}{c[:18]}\n", style="#94A3B8")

        task_type = s.get("task_type", "")
        if task_type:
            row("task", task_type, "#8B5CF6")

        return t


class StepLog(RichLog):
    """Center panel — chronological execution timeline."""

    DEFAULT_CSS = """
    StepLog {
        border: tall #1E293B;
        background: #0B0F14;
        width: 1fr;
        padding: 0 1;
        scrollbar-color: #1E293B #0B0F14;
        scrollbar-color-hover: #4D7CFE #0B0F14;
    }
    """

    _KINDS: dict[str, tuple[str, str]] = {
        "start": ("◎ ", "#94A3B8"),
        "think": ("→ ", "#4D7CFE"),
        "act": ("✓ ", "#3BE8FF"),
        "error": ("✗ ", "#EF4444"),
        "validate": ("↳ ", "#8B5CF6"),
        "done": ("◆ ", "#22C55E"),
        "inject": ("⊕ ", "#F59E0B"),
    }

    def add(self, step: int, kind: str, text: str) -> None:
        icon, color = self._KINDS.get(kind, ("· ", "#94A3B8"))
        line = Text()
        line.append(f"[{step:02d}]", style="#1E293B")
        line.append(f" {icon}", style=color)
        msg_style = "#94A3B8" if kind == "validate" else "#E2E8F0"
        line.append(text, style=msg_style)
        self.write(line)


class MemoryPanel(Widget):
    """Right panel — working memory key-value list."""

    DEFAULT_CSS = """
    MemoryPanel {
        height: 1fr;
        border: tall #1E293B;
        background: #0B0F14;
        padding: 0 1;
        overflow-y: auto;
        scrollbar-color: #1E293B #0B0F14;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._memory: dict[str, Any] = {}

    def upsert(self, key: str, value: Any) -> None:
        self._memory[key] = value
        self.refresh()

    def replace(self, memory: dict[str, Any]) -> None:
        self._memory = dict(memory)
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(" MEMORY\n", style="#94A3B8")
        t.append("─" * 22 + "\n", style="#1E293B")

        if not self._memory:
            t.append(" —\n", style="#1E293B")
            return t

        for key, val in list(self._memory.items())[-24:]:
            val_str = str(val)
            if len(val_str) > 28:
                val_str = val_str[:25] + "…"
            t.append(f" {key[:14]:<15}", style="#8B5CF6")
            t.append(f"{val_str}\n", style="#94A3B8")

        return t


class ContextPanel(Widget):
    """Bottom-left — extracted context items and open questions."""

    DEFAULT_CSS = """
    ContextPanel {
        height: auto;
        min-height: 7;
        border: tall #1E293B;
        background: #0B0F14;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._items: list[str] = []
        self._questions: list[str] = []

    def update(self, items: list[str], questions: list[str]) -> None:
        self._items = items
        self._questions = questions
        self.refresh()

    def render(self) -> Text:
        t = Text()

        if self._items:
            t.append(" CONTEXT\n", style="#94A3B8")
            for item in self._items[:5]:
                bracket_end = item.find("]")
                if bracket_end > 0:
                    tag = item[: bracket_end + 1]
                    rest = item[bracket_end + 1 :].strip()[:22]
                    t.append(f"  {tag}", style="#4D7CFE")
                    t.append(f" {rest}\n", style="#94A3B8")
                else:
                    t.append(f"  {item[:30]}\n", style="#94A3B8")

        if self._questions:
            t.append(" OPEN\n", style="#94A3B8")
            for q in self._questions[:3]:
                t.append("  ? ", style="#F59E0B")
                t.append(f"{q[:26]}\n", style="#94A3B8")

        if not self._items and not self._questions:
            t.append(" —\n", style="#1E293B")

        return t


class ArtifactsBar(Widget):
    """Single-line artifacts strip at the bottom of the right column."""

    DEFAULT_CSS = """
    ArtifactsBar {
        height: 3;
        border: tall #1E293B;
        background: #0B0F14;
        padding: 0 1;
        overflow-x: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._artifacts: list[dict[str, str]] = []

    def add_artifact(self, type_: str, name: str, description: str = "") -> None:
        self._artifacts.append(
            {"type": type_, "name": name, "description": description}
        )
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(" ARTIFACTS\n", style="#94A3B8")

        if not self._artifacts:
            t.append(" —", style="#1E293B")
            return t

        for art in self._artifacts[-6:]:
            t.append(" ◆ ", style="#3BE8FF")
            t.append(f"{art['type']}:", style="#4D7CFE")
            t.append(f"{art['name']}  ", style="#E2E8F0")

        return t


class CommandBar(Widget):
    """Bottom prompt / inject / goals input bar."""

    DEFAULT_CSS = """
    CommandBar {
        dock: bottom;
        height: 3;
        background: #0D1219;
        layout: horizontal;
        padding: 0 1;
    }
    #arc-mode {
        width: 12;
        height: 1;
        margin-top: 1;
        color: #4D7CFE;
    }
    #arc-input {
        width: 1fr;
        margin-top: 0;
        background: #0D1219;
        color: #E2E8F0;
        border: none;
    }
    #arc-hints {
        width: 38;
        height: 1;
        margin-top: 1;
        color: #1E293B;
        text-align: right;
    }
    """

    mode: reactive[str] = reactive("prompt")

    def compose(self) -> ComposeResult:
        yield Static("› prompt", id="arc-mode")
        yield Input(placeholder="type a query and press enter…", id="arc-input")
        yield Static("^P pause  ^I inject  ^G goals  ^X abort", id="arc-hints")

    def watch_mode(self, mode: str) -> None:
        colors = {"prompt": "#4D7CFE", "inject": "#3BE8FF", "goals": "#8B5CF6"}
        color = colors.get(mode, "#4D7CFE")
        placeholders = {
            "prompt": "type a query and press enter…",
            "inject": "key=value  (inject into memory)",
            "goals": "goal1, goal2, …  (overwrite goals)",
        }
        hints = {
            "prompt": "[#475569]^P pause  ^I inject  ^G goals  ^X abort[/]",
            "inject": "[#3BE8FF]key=value[/]  [#475569][ESC] cancel[/]",
            "goals": "[#8B5CF6]goal1, goal2, …[/]  [#475569][ESC] cancel[/]",
        }
        self.query_one("#arc-mode", Static).update(f"[{color}]› {mode}[/]")
        self.query_one("#arc-hints", Static).update(hints.get(mode, ""))
        inp = self.query_one("#arc-input", Input)
        inp.placeholder = placeholders.get(mode, "")


# ── Main App ──────────────────────────────────────────────────────────────────


class ArcApp(App):
    """ARC — Action & Reasoning Core TUI."""

    CSS = """
    Screen {
        background: #0B0F14;
        color: #E2E8F0;
    }
    #arc-body {
        height: 1fr;
    }
    #col-left {
        width: 26;
    }
    #col-right {
        width: 28;
    }
    """

    # Use ctrl+ prefixes so bindings don't fire while typing in the Input.
    BINDINGS = [
        Binding("ctrl+p", "pause_agent", "Pause", show=False),
        Binding("ctrl+r", "resume_agent", "Resume", show=False),
        Binding("ctrl+x", "abort_agent", "Abort", show=False),
        Binding("ctrl+i", "mode_inject", "Inject", show=False),
        Binding("ctrl+g", "mode_goals", "Goals", show=False),
        Binding("escape", "mode_prompt", "Prompt", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, agent_factory: Callable[[], Any] | None = None) -> None:
        super().__init__()
        self._agent_factory = agent_factory
        self._agent: Any = None  # live reference — used for memory injection
        self._paused = threading.Event()
        self._abort = threading.Event()
        self._running = False

    # ── layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield ArcHeader()
        with Horizontal(id="arc-body"):
            with Vertical(id="col-left"):
                yield StatePanel()
                yield ContextPanel()
            yield StepLog(highlight=False, markup=False, wrap=False)
            with Vertical(id="col-right"):
                yield MemoryPanel()
                yield ArtifactsBar()
        yield CommandBar()

    def on_mount(self) -> None:
        self.set_interval(1.0, self._tick)
        self.push_screen(
            LaunchScreen(has_factory=self._agent_factory is not None),
            self._on_launch_config,
        )

    def _on_launch_config(self, config: dict[str, Any] | None) -> None:
        """Callback from LaunchScreen.dismiss() — build factory + kick off run."""
        if config is None:
            self.exit()
            return

        # Build factory from launch screen config when none was pre-supplied.
        if self._agent_factory is None:
            model_path = config.get("model_path", "./models/model.gguf")
            n_ctx = config.get("n_ctx", 6000)
            n_gpu_layers = config.get("n_gpu_layers", 0)
            chat_format = config.get("chat_format")

            def _factory(
                _path=model_path,
                _ctx=n_ctx,
                _gpu=n_gpu_layers,
                _fmt=chat_format,
            ):
                from src.agent import Agent
                from src.llama_runtime import ModelSource, RuntimeOptions

                return Agent(
                    ModelSource(model_path=_path),
                    RuntimeOptions(
                        n_ctx=_ctx,
                        n_gpu_layers=_gpu,
                        chat_format=_fmt,
                        idle_unload_seconds=120,
                    ),
                )

            self._agent_factory = _factory

        query = config.get("query", "")
        if query:
            self._start(query)

        # Restore input focus after launch screen dismissal.
        self.query_one("#arc-input", Input).focus()

    def _tick(self) -> None:
        """Refresh header every second for the elapsed timer."""
        hdr = self.query_one(ArcHeader)
        if hdr._start_time is not None and hdr.status == "running":
            hdr.refresh()

    # ── input ─────────────────────────────────────────────────────────────────

    @on(Input.Submitted, "#arc-input")
    def handle_submit(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return

        bar = self.query_one(CommandBar)
        mode = bar.mode

        if mode == "prompt":
            if self._running:
                self.query_one(StepLog).add(0, "validate", "agent already running")
                return
            event.input.clear()
            self._start(value)
            return

        if mode == "inject":
            if "=" in value:
                k, v = value.split("=", 1)
                self._inject_memory(k.strip(), v.strip())
            event.input.clear()
            bar.mode = "prompt"
            return

        if mode == "goals":
            goals = [g.strip() for g in value.split(",") if g.strip()]
            self._set_goals(goals)
            event.input.clear()
            bar.mode = "prompt"
            return

    # ── key actions ───────────────────────────────────────────────────────────

    def action_pause_agent(self) -> None:
        if not self._running:
            return
        self._paused.set()
        self.query_one(ArcHeader).status = "paused"
        self.query_one(StepLog).add(0, "validate", "paused by user")

    def action_resume_agent(self) -> None:
        self._paused.clear()
        if self._running:
            self.query_one(ArcHeader).status = "running"
            self.query_one(StepLog).add(0, "start", "resumed")

    def action_abort_agent(self) -> None:
        self._abort.set()
        self._paused.clear()
        self._running = False
        self.query_one(ArcHeader).status = "idle"
        self.query_one(StepLog).add(0, "error", "aborted by user")

    def action_mode_inject(self) -> None:
        self.query_one(CommandBar).mode = "inject"
        self.query_one("#arc-input", Input).focus()

    def action_mode_goals(self) -> None:
        self.query_one(CommandBar).mode = "goals"
        self.query_one("#arc-input", Input).focus()

    def action_mode_prompt(self) -> None:
        self.query_one(CommandBar).mode = "prompt"
        self.query_one("#arc-input", Input).focus()

    # ── agent runner ──────────────────────────────────────────────────────────

    def _start(self, query: str) -> None:
        self._abort.clear()
        self._paused.clear()
        self._running = True

        hdr = self.query_one(ArcHeader)
        hdr.status = "running"
        hdr._start_time = time.time()
        hdr.step = 0
        hdr.artifact_count = 0

        log = self.query_one(StepLog)
        log.clear()
        log.add(0, "start", f"query: {query}")

        self._run_worker(query)

    @work(thread=True)
    def _run_worker(self, query: str) -> None:
        """Run the agent in a background thread, posting events back to the TUI."""
        try:
            if self._agent_factory is None:
                raise RuntimeError(
                    "No agent factory configured. "
                    "Pass agent_factory=... to run() or use the launch screen."
                )

            agent = self._agent_factory()
            self._agent = agent  # keep reference so _inject_memory can reach live state

            def on_event(event: Any) -> None:
                self.post_message(AgentEvent(event))

            agent.event_bus.subscribe(on_event)

            agent.run(query)

        except Exception as exc:
            self.post_message(
                AgentEvent(
                    Event(
                        type=EventType.AGENT_ERROR,
                        payload={"error": str(exc)},
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            )
        finally:
            self._running = False
            self._agent = None

    # ── event routing ─────────────────────────────────────────────────────────

    def on_agent_event(self, msg: AgentEvent) -> None:
        event = msg.event
        et = event.type.name
        p = event.payload
        step = event.step or 0
        run_id = event.run_id
        duration = event.duration_ms
        error = event.error

        hdr = self.query_one(ArcHeader)
        log = self.query_one(StepLog)
        state = self.query_one(StatePanel)
        mem = self.query_one(MemoryPanel)
        ctx = self.query_one(ContextPanel)
        arts = self.query_one(ArtifactsBar)

        if et == "AGENT_STARTED":
            log.add(0, "start", "agent started")

        elif et == "INTENT_EXTRACTED":
            intent = p.get("intent", "")
            goals = p.get("goals", [])
            log.add(0, "think", f"intent: {intent[:50]}")
            state.update_state({"intent": intent, "goals": goals, "status": "new"})

        elif et == "STEP_STARTED":
            hdr.step = step
            s = p.get("state", {})
            state.update_state(s)
            ctx.update(s.get("context_items", []), s.get("open_questions", []))
            mem.replace(s.get("working_memory", {}))

        elif et == "ACTION_PLANNED":
            action = p.get("action", {})
            tool = action.get("tool", "?")
            inp = str(action.get("input", ""))[:50]
            log.add(hdr.step, "think", f"{tool}  {inp}")

        elif et == "ACTION_EXECUTED":
            success = p.get("success", True)
            summary = p.get("summary", "")[:70]
            ms = f"{duration:.0f}ms" if duration is not None else ""
            log.add(step, "act" if success else "error", f"{summary} {ms}".strip())

        elif et == "VALIDATION_DONE":
            status = p.get("status", "")
            reason = p.get("reason", "")[:60]
            ms = f"{duration:.0f}ms" if duration is not None else ""
            log.add(step, "validate", f"{status}: {reason} {ms}".strip())

        elif et == "MEMORY_UPDATED":
            key = p.get("key", "")
            value = p.get("value", "")
            mem.upsert(key, value)
            # Artifact keys are stored as "artifact_N" → track in header + bar
            if key.startswith("artifact_") and isinstance(value, list):
                hdr.artifact_count += len(value)
                for name in value:
                    arts.add_artifact("artifact", name)

        elif et == "AGENT_DONE":
            hdr.status = "done"
            s = p.get("state", {})
            state.update_state({**s, "status": "done"})
            mem.replace(s.get("working_memory", {}))
            log.add(hdr.step, "done", "agent finished")
            self._running = False

        elif et == "AGENT_ERROR":
            hdr.status = "error"
            log.add(hdr.step, "error", p.get("error", "unknown error")[:80])

        elif et == "THINKING_STARTED":
            log.add(step, "think", "thinking started")

        elif et == "THINKING_ENDED":
            action = p.get("action", {})
            label = action.get("tool", "unknown")
            ms = f"{duration:.0f}ms" if duration is not None else ""
            log.add(step, "think", f"planned {label} {ms}".strip())

        elif et == "VALIDATION_STARTED":
            log.add(step, "validate", "validation started")

        elif et == "TOOL_ERROR":
            log.add(step, "error", error or p.get("message", "tool error"))

        elif et == "WARNING":
            log.add(step, "validate", p.get("message", "warning"))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _inject_memory(self, key: str, value: str) -> None:
        """Inject a key/value into the UI panel AND the live agent state."""
        self.query_one(MemoryPanel).upsert(key, value)
        self.query_one(StepLog).add(
            self.query_one(ArcHeader).step, "inject", f"{key} = {value}"
        )
        # Propagate to running agent state if accessible.
        agent = self._agent
        if agent is not None:
            live_state = getattr(agent, "_current_state", None)
            if live_state is not None:
                live_state.remember(key, value)

    def _set_goals(self, goals: list[str]) -> None:
        """Overwrite goals in the UI panel AND the live agent state."""
        state_panel = self.query_one(StatePanel)
        state_panel.update_state({**state_panel._state, "goals": goals})
        self.query_one(StepLog).add(
            self.query_one(ArcHeader).step, "inject", f"goals → {goals}"
        )
        # Propagate to running agent state if accessible.
        agent = self._agent
        if agent is not None:
            live_state = getattr(agent, "_current_state", None)
            if live_state is not None:
                live_state.goals = list(goals)


# ── entry point ───────────────────────────────────────────────────────────────


def run(agent_factory: Callable[[], Any] | None = None) -> None:
    """Launch the ARC TUI.

    Args:
        agent_factory: zero-arg callable returning a configured Agent.
                       If None, the launch screen will collect model path +
                       params and construct the agent automatically.

    Example::

        from src.agent import Agent
        from src.llama_runtime import ModelSource

        run(agent_factory=lambda: Agent(ModelSource.from_path("model.gguf")))
    """
    ArcApp(agent_factory=agent_factory).run()


if __name__ == "__main__":
    # Default: show full launch screen (no pre-baked factory).
    run()
