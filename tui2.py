from __future__ import annotations

"""Testing-oriented menu TUI for the Agent.

Features
- Menu-style navigation using questionary.
- Beautiful rich output with panels, tables, and live status.
- Per-run event tracing from Agent.event_bus.
- Hang detection via timeout.
- Verbose/debug mode for event payloads and tracebacks.
- Run history with quick inspection of the last execution.
- Batch test mode for multiple queries.

Usage example
------------
from src.llama_runtime import ModelSource
from src.agent import Agent
from tui.agent_testing_tui import run_tui

run_tui(lambda: Agent(ModelSource.from_path("model.gguf")))
"""

from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable, Optional
import json
import threading
import time
import traceback

import questionary
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from src.events import Event, EventType


console = Console()


@dataclass
class RunRecord:
    run_id: str
    query: str
    started_at: datetime
    ended_at: datetime | None = None
    status: str = "running"  # running | done | error | timeout
    duration_s: float | None = None
    result: Any = None
    error: str | None = None
    traceback_text: str | None = None
    events: list[Event] = field(default_factory=list)

    @property
    def event_count(self) -> int:
        return len(self.events)


@dataclass
class TUISettings:
    timeout_s: float = 60.0
    show_payloads: bool = True
    auto_clear: bool = False
    save_logs: bool = True
    log_dir: Path = Path("./run_logs")


class AgentRunMonitor:
    """Subscribe to an agent event bus and keep a live snapshot of a run."""

    def __init__(self, record: RunRecord, settings: TUISettings):
        self.record = record
        self.settings = settings
        self._lock = threading.Lock()
        self._last_event: Event | None = None
        self._step = 0
        self._start_time = time.time()
        self._status_line = "starting"

    def __call__(self, event: Event) -> None:
        with self._lock:
            self.record.events.append(event)
            self._last_event = event
            if event.step is not None:
                self._step = max(self._step, event.step)

            if event.type == EventType.AGENT_STARTED:
                self._status_line = "agent started"
            elif event.type == EventType.INTENT_EXTRACTED:
                self._status_line = "intent extracted"
            elif event.type == EventType.STEP_STARTED:
                self._status_line = f"step {event.step} started"
            elif event.type == EventType.THINKING_STARTED:
                self._status_line = f"step {event.step} thinking"
            elif event.type == EventType.THINKING_ENDED:
                self._status_line = f"step {event.step} planned action"
            elif event.type == EventType.ACTION_EXECUTING:
                self._status_line = f"step {event.step} executing action"
            elif event.type == EventType.ACTION_EXECUTED:
                self._status_line = f"step {event.step} action executed"
            elif event.type == EventType.VALIDATION_DONE:
                self._status_line = f"step {event.step} validated"
            elif event.type == EventType.AGENT_DONE:
                self._status_line = "done"
                self.record.status = "done"
            elif event.type == EventType.AGENT_ERROR:
                self._status_line = "error"
                self.record.status = "error"
                self.record.error = event.error
                self.record.traceback_text = event.payload.get("traceback")

            if event.error and self.record.error is None:
                self.record.error = event.error

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._start_time
            return {
                "status": self.record.status,
                "step": self._step,
                "elapsed": elapsed,
                "last_event": self._last_event,
                "status_line": self._status_line,
                "events": list(self.record.events),
            }


class AgentTestingTUI:
    def __init__(
        self, agent_factory: Callable[[], Any], settings: TUISettings | None = None
    ):
        self.agent_factory = agent_factory
        self.settings = settings or TUISettings()
        self.history: list[RunRecord] = []
        self._last_agent: Any | None = None

    def _new_agent(self):
        agent = self.agent_factory()
        self._last_agent = agent
        return agent

    def _banner(self) -> Panel:
        title = Text("Agent Testing TUI", style="bold magenta")
        body = Text(
            "Run queries, inspect events, detect hangs, and get verbose diagnostics.",
            style="white",
        )
        return Panel(
            Align.center(Group(title, body)),
            border_style="magenta",
            padding=(1, 2),
        )

    def _menu(self) -> str:
        choices = [
            {"name": "Run one query", "value": "run"},
            {"name": "Batch test queries", "value": "batch"},
            {"name": "Show last run", "value": "last"},
            {"name": "Show run history", "value": "history"},
            {"name": "Settings", "value": "settings"},
            {"name": "Quit", "value": "quit"},
        ]
        return (
            questionary.select(
                "Choose an action:",
                choices=choices,
                use_shortcuts=True,
                use_arrow_keys=True,
            ).ask()
            or "quit"
        )

    def _prompt_query(self) -> str | None:
        # multiline=True in questionary often feels like a hang because Enter inserts a newline.
        # Use a normal prompt for fast testing UX.
        return questionary.text(
            "Enter a test query:",
        ).ask()

    def _prompt_batch(self) -> list[str]:
        console.print(
            "[dim]Finish batch input with CTRL+D (Linux/macOS) or CTRL+Z then Enter (Windows).[/dim]"
        )
        raw = questionary.text(
            "Enter one query per line (blank lines are ignored):",
            multiline=True,
        ).ask()
        if not raw:
            return []
        return [line.strip() for line in raw.splitlines() if line.strip()]

    def _settings_menu(self) -> None:
        while True:
            table = Table(title="Current Settings", show_lines=True)
            table.add_column("Key")
            table.add_column("Value")
            table.add_row("timeout_s", str(self.settings.timeout_s))
            table.add_row("show_payloads", str(self.settings.show_payloads))
            table.add_row("auto_clear", str(self.settings.auto_clear))
            table.add_row("save_logs", str(self.settings.save_logs))
            table.add_row("log_dir", str(self.settings.log_dir))
            console.print(table)

            choice = questionary.select(
                "Settings menu:",
                choices=[
                    {"name": "Set timeout", "value": "timeout"},
                    {"name": "Toggle payload visibility", "value": "payloads"},
                    {"name": "Toggle auto clear", "value": "clear"},
                    {"name": "Toggle log saving", "value": "logs"},
                    {"name": "Change log directory", "value": "dir"},
                    {"name": "Back", "value": "back"},
                ],
            ).ask()

            if choice in (None, "back"):
                return
            if choice == "timeout":
                value = questionary.text(
                    "Timeout seconds:", default=str(self.settings.timeout_s)
                ).ask()
                if value:
                    try:
                        self.settings.timeout_s = float(value)
                    except ValueError:
                        console.print("[red]Invalid timeout value.[/red]")
            elif choice == "payloads":
                self.settings.show_payloads = not self.settings.show_payloads
            elif choice == "clear":
                self.settings.auto_clear = not self.settings.auto_clear
            elif choice == "logs":
                self.settings.save_logs = not self.settings.save_logs
            elif choice == "dir":
                value = questionary.text(
                    "Log directory:",
                    default=str(self.settings.log_dir),
                ).ask()
                if value:
                    self.settings.log_dir = Path(value).expanduser().resolve()

    def _write_run_log(self, record: RunRecord) -> None:
        if not self.settings.save_logs:
            return
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.log_dir / f"{record.run_id}.json"
        payload = {
            "run_id": record.run_id,
            "query": record.query,
            "started_at": record.started_at.isoformat(),
            "ended_at": record.ended_at.isoformat() if record.ended_at else None,
            "status": record.status,
            "duration_s": record.duration_s,
            "error": record.error,
            "traceback": record.traceback_text,
            "events": [
                {
                    "type": e.type.name,
                    "ts": e.ts,
                    "run_id": e.run_id,
                    "step": e.step,
                    "error": e.error,
                    "payload": e.payload,
                    "duration_ms": e.duration_ms,
                    "source": e.source,
                }
                for e in record.events
            ],
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _render_run_view(self, record: RunRecord, monitor: AgentRunMonitor) -> Panel:
        snap = monitor.snapshot()
        last_event = snap["last_event"]

        info = Table.grid(padding=(0, 2))
        info.add_column(justify="right", style="bold cyan")
        info.add_column()
        info.add_row("Run ID", record.run_id)
        info.add_row("Status", f"[bold]{record.status}[/bold]")
        info.add_row("Step", str(snap["step"]))
        info.add_row("Elapsed", f"{snap['elapsed']:.1f}s")
        info.add_row("Events", str(record.event_count))
        info.add_row("Live", snap["status_line"])

        recent_events = list(record.events[-8:])
        if recent_events:
            lines = []
            for ev in recent_events:
                line = f"[{ev.type.name}]"
                if ev.step is not None:
                    line += f" step={ev.step}"
                if ev.duration_ms is not None:
                    line += f" {ev.duration_ms:.1f}ms"
                if ev.error:
                    line += f" ERROR={ev.error}"
                lines.append(line)
            event_block = "\\n".join(lines)
        else:
            event_block = "No events yet."
        if last_event is not None:
            event_block = f"{last_event.type.name}\n{json.dumps(last_event.payload, indent=2, ensure_ascii=False)}"

        body = Group(
            info,
            Rule(),
            Text("Recent events", style="bold"),
            Panel(event_block, border_style="blue"),
        )
        return Panel(body, title="Run Monitor", border_style="green")

    def _print_event(self, event: Event) -> None:
        style_map = {
            EventType.AGENT_STARTED: "bold green",
            EventType.INTENT_EXTRACTED: "cyan",
            EventType.STEP_STARTED: "yellow",
            EventType.CONTEXT_BUILT: "blue",
            EventType.THINKING_STARTED: "magenta",
            EventType.THINKING_ENDED: "magenta",
            EventType.ACTION_PLANNED: "bright_cyan",
            EventType.ACTION_EXECUTING: "yellow",
            EventType.ACTION_EXECUTED: "green",
            EventType.VALIDATION_STARTED: "yellow",
            EventType.VALIDATION_DONE: "green",
            EventType.AGENT_DONE: "bold green",
            EventType.AGENT_ERROR: "bold red",
            EventType.WARNING: "bright_yellow",
        }
        style = style_map.get(event.type, "white")
        header = f"[{event.type.name}]"
        meta = f"step={event.step}  run={event.run_id[:8]}"
        if event.duration_ms is not None:
            meta += f"  {event.duration_ms:.1f}ms"
        if event.error:
            meta += f"  error={event.error}"
        console.print(f"{header} {meta}", style=style)

        if self.settings.show_payloads and event.payload:
            console.print(Pretty(event.payload, expand_all=False))

    def _run_agent_with_monitor(self, query: str) -> RunRecord:
        agent = self._new_agent()
        run_id = uuid4_hex()
        record = RunRecord(run_id=run_id, query=query, started_at=datetime.now())
        monitor = AgentRunMonitor(record, self.settings)

        # Subscribe to this agent's event bus.
        agent.event_bus.subscribe(monitor)
        # Avoid direct terminal writes during Live rendering.
        # Rich Live + external prints can freeze / corrupt the UI on some terminals.
        # Events are instead rendered through the live monitor.
        # agent.event_bus.subscribe(self._print_event)

        def _runner() -> Any:
            return agent.run(query)

        with Live(
            self._render_run_view(record, monitor),
            console=console,
            refresh_per_second=8,
            transient=False,
        ) as live:
            start = time.time()
            # Use a dedicated thread executor for the agent.
            # Small local models can fully block the main thread otherwise.
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="agent-runner"
            ) as pool:
                future = pool.submit(_runner)
                try:
                    while True:
                        try:
                            result = future.result(timeout=0.2)
                            record.result = result
                            break
                        except FutureTimeoutError:
                            elapsed = time.time() - start
                            live.update(self._render_run_view(record, monitor))
                            if elapsed > self.settings.timeout_s:
                                record.status = "timeout"
                                record.error = (
                                    f"Timed out after {self.settings.timeout_s:.1f}s"
                                )
                                future.cancel()
                                break
                        except Exception as exc:
                            record.status = "error"
                            record.error = f"{type(exc).__name__}: {exc}"
                            record.traceback_text = traceback.format_exc()
                            break
                finally:
                    record.ended_at = datetime.now()
                    record.duration_s = (
                        record.ended_at - record.started_at
                    ).total_seconds()
                    live.update(self._render_run_view(record, monitor))

        self.history.append(record)
        self._write_run_log(record)
        self._show_run_summary(record)
        return record

    def _show_run_summary(self, record: RunRecord) -> None:
        console.print()
        if record.status == "done":
            console.print(
                Panel.fit("Run completed successfully.", border_style="green")
            )
        elif record.status == "timeout":
            console.print(
                Panel.fit(f"Run hung / timed out: {record.error}", border_style="red")
            )
        else:
            console.print(
                Panel.fit(
                    f"Run failed: {record.error or 'unknown error'}", border_style="red"
                )
            )

        summary = Table(title="Run Summary", show_lines=True)
        summary.add_column("Field", style="bold cyan")
        summary.add_column("Value")
        summary.add_row("Run ID", record.run_id)
        summary.add_row("Status", record.status)
        summary.add_row(
            "Duration",
            f"{record.duration_s:.2f}s" if record.duration_s is not None else "-",
        )
        summary.add_row("Events", str(record.event_count))
        summary.add_row("Query", record.query)
        console.print(summary)

        if record.traceback_text:
            console.print(
                Panel(record.traceback_text, title="Traceback", border_style="red")
            )

    def _show_history(self) -> None:
        if not self.history:
            console.print(Panel.fit("No runs yet.", border_style="yellow"))
            return

        table = Table(title="Run History", show_lines=True)
        table.add_column("#", justify="right")
        table.add_column("Status")
        table.add_column("Duration")
        table.add_column("Events", justify="right")
        table.add_column("Query")
        table.add_column("Run ID")

        for idx, rec in enumerate(reversed(self.history[-20:]), start=1):
            table.add_row(
                str(idx),
                rec.status,
                f"{rec.duration_s:.2f}s" if rec.duration_s is not None else "-",
                str(rec.event_count),
                truncate(rec.query, 48),
                rec.run_id[:10],
            )
        console.print(table)

    def _show_last_run(self) -> None:
        if not self.history:
            console.print(Panel.fit("No previous run.", border_style="yellow"))
            return
        record = self.history[-1]
        self._show_run_summary(record)

    def run(self) -> None:
        console.clear()
        console.print(self._banner())

        while True:
            choice = self._menu()
            if choice == "quit":
                console.print("[bold cyan]Bye.[/bold cyan]")
                return
            if choice == "settings":
                self._settings_menu()
                continue
            if choice == "history":
                self._show_history()
                continue
            if choice == "last":
                self._show_last_run()
                continue
            if choice == "run":
                query = self._prompt_query()
                if query:
                    self._run_agent_with_monitor(query)
                continue
            if choice == "batch":
                queries = self._prompt_batch()
                if not queries:
                    continue
                self._run_batch(queries)
                continue

    def _run_batch(self, queries: list[str]) -> None:
        console.print(
            Panel.fit(f"Running {len(queries)} test queries...", border_style="cyan")
        )
        batch_table = Table(title="Batch Results", show_lines=True)
        batch_table.add_column("#", justify="right")
        batch_table.add_column("Status")
        batch_table.add_column("Duration")
        batch_table.add_column("Events", justify="right")
        batch_table.add_column("Query")

        for idx, query in enumerate(queries, start=1):
            console.print(Rule(f"Test {idx}/{len(queries)}"))
            record = self._run_agent_with_monitor(query)
            batch_table.add_row(
                str(idx),
                record.status,
                f"{record.duration_s:.2f}s" if record.duration_s is not None else "-",
                str(record.event_count),
                truncate(query, 42),
            )

        console.print(batch_table)


def truncate(text: str, max_len: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def uuid4_hex() -> str:
    import uuid

    return uuid.uuid4().hex


def run_tui(agent_factory: Callable[[], Any]) -> None:
    """Entry point for the testing TUI."""
    tui = AgentTestingTUI(agent_factory)
    tui.run()


if __name__ == "__main__":
    console.print(
        Panel.fit(
            "Import this module and call run_tui(lambda: Agent(...))",
            title="Agent Testing TUI",
            border_style="magenta",
        )
    )

if __name__ == "__main__":
    """
    console.print(
        Panel.fit(
            "Import this module and call run_tui(lambda: Agent(...))",
            title="Agent Testing TUI",
            border_style="magenta",
        )
    )
    """

    from src.llama_runtime import ModelSource, RuntimeOptions
    from src.agent import Agent

    run_tui(
        lambda: Agent(
            ModelSource(model_path="./models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"),
            RuntimeOptions(
                n_ctx=6000,
                n_gpu_layers=0,  # raise this for GPU offload
                idle_unload_seconds=120,
                chat_format=None,  # set if your model needs it
            ),
        )
    )
