import asyncio
from typing import Any, Callable

import questionary
import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.agent import Agent, AgentConfig
from src.events import Event, EventType
from src.cli.theme import arc_status
from src.util.models import list_models, resolve_model_path

console = Console()


def _event_type(name: str):
    return getattr(EventType, name, None)


EVENT_AGENT_STARTED = _event_type("AGENT_STARTED")
EVENT_INTENT_EXTRACTED = _event_type("INTENT_EXTRACTED")
EVENT_STEP_STARTED = _event_type("STEP_STARTED")
EVENT_CONTEXT_BUILT = _event_type("CONTEXT_BUILT")
EVENT_THINKING_STARTED = _event_type("THINKING_STARTED")
EVENT_THINKING_ENDED = _event_type("THINKING_ENDED")
EVENT_ACTION_PLANNED = _event_type("ACTION_PLANNED")
EVENT_ACTION_STARTED = _event_type("ACTION_STARTED")
EVENT_ACTION_FINISHED = _event_type("ACTION_FINISHED")
EVENT_ACTION_EXECUTING = _event_type("ACTION_EXECUTING")
EVENT_ACTION_EXECUTED = _event_type("ACTION_EXECUTED")
EVENT_VALIDATION_STARTED = _event_type("VALIDATION_STARTED")
EVENT_VALIDATION_FINISHED = _event_type("VALIDATION_FINISHED")
EVENT_VALIDATION_DONE = _event_type("VALIDATION_DONE")
EVENT_SYNTHESIS_STARTED = _event_type("SYNTHESIS_STARTED")
EVENT_SYNTHESIS_FINISHED = _event_type("SYNTHESIS_FINISHED")
EVENT_STATE_UPDATED = _event_type("STATE_UPDATED")
EVENT_AGENT_DONE = _event_type("AGENT_DONE")
EVENT_AGENT_ERROR = _event_type("AGENT_ERROR")
EVENT_TOOL_ERROR = _event_type("TOOL_ERROR")
EVENT_WARNING = _event_type("WARNING")
EVENT_MEMORY_UPDATED = _event_type("MEMORY_UPDATED")
EVENT_REASONING_STARTED = _event_type("REASONING_STARTED")
EVENT_REASONING_CHUNK = _event_type("REASONING_CHUNK")
EVENT_REASONING_FINISHED = _event_type("REASONING_FINISHED")


def _is_event_type(event: Event, *candidates: Any) -> bool:
    return any(
        candidate is not None and event.type == candidate for candidate in candidates
    )


def _obj_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()  # type: ignore[attr-defined]
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {k: v for k, v in vars(value).items() if not k.startswith("_")}
        except Exception:
            pass

    return {}


def _fmt(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value}"
    if isinstance(value, dict):
        if not value:
            return "{}"
        parts = []
        for k, v in list(value.items())[:6]:
            parts.append(f"{k}={_fmt(v)}")
        suffix = "…" if len(value) > 6 else ""
        return ", ".join(parts) + suffix
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if not items:
            return "[]"
        parts = [_fmt(v) for v in items[:6]]
        suffix = "…" if len(items) > 6 else ""
        return ", ".join(parts) + suffix
    return str(value)


def _short(text: Any, limit: int = 180) -> str:
    s = _fmt(text)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _bullets(items: Any, limit: int = 8) -> str:
    if not items:
        return "-"
    if not isinstance(items, (list, tuple, set)):
        return _fmt(items)
    values = list(items)
    if not values:
        return "-"
    return "\n".join(f"• {_short(v, 120)}" for v in values[:limit])


def _state_payload(event: Event) -> dict[str, Any]:
    state = event.payload.get("state", {})
    if isinstance(state, dict):
        return state
    return _obj_to_dict(state)


def _action_payload(event: Event) -> dict[str, Any]:
    action = event.payload.get("action", {})
    if isinstance(action, dict):
        return action
    return _obj_to_dict(action)


def _result_payload(event: Event) -> dict[str, Any]:
    result = event.payload.get("result", {})
    if isinstance(result, dict):
        return result
    return _obj_to_dict(result)


def _validation_payload(event: Event) -> dict[str, Any]:
    validation = event.payload.get("validation", {})
    if isinstance(validation, dict):
        return validation
    return _obj_to_dict(validation)


def _answer_payload(event: Event) -> dict[str, Any]:
    answer = event.payload.get("answer", event.payload.get("output", {}))
    if isinstance(answer, dict):
        return answer
    return _obj_to_dict(answer)


def _duration_text(duration_ms: float | None) -> str:
    if duration_ms is None:
        return "-"
    return f"{duration_ms:.1f} ms"


def _panel(
    title: str,
    border_style: str,
    rows: list[tuple[str, Any]],
    *,
    subtitle: str | None = None,
) -> None:
    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False, expand=True)
    table.add_column("Key", style="bold")
    table.add_column("Value", overflow="fold")
    for key, value in rows:
        if value is None or value == "" or value == [] or value == {}:
            continue
        table.add_row(str(key), value if isinstance(value, str) else _fmt(value))
    console.print(
        Panel(table, border_style=border_style, title=title, subtitle=subtitle)
    )


def _validation_color(status: str) -> str:
    status = status.lower().strip()
    return {
        "done": "green",
        "complete": "green",
        "present": "cyan",
        "replan": "yellow",
        "continue": "blue",
        "working": "blue",
        "error": "red",
        "failed": "red",
    }.get(status, "white")


def build_cli_event_handler(*, debug: bool = False) -> Callable[[Event], None]:
    # Live-updating "thinking..." panels, keyed by (run_id, role) so multiple
    # roles (or overlapping runs, in theory) don't clobber each other's
    # display. REASONING_CHUNK events arrive one token at a time; batching
    # them all into a single Panel re-render via Live avoids flooding the
    # terminal with one panel per token.
    reasoning_live: dict[tuple[str, str], Live] = {}
    reasoning_text: dict[tuple[str, str], str] = {}

    def _reasoning_panel(role: str, text: str, *, done: bool) -> Panel:
        body = text.strip() or "…"
        label = role.capitalize() if role else "Model"
        return Panel(
            Text(body, style="italic" if done else "dim italic"),
            title=f"{label} reasoning" if done else f"{label} is thinking…",
            border_style="dim magenta" if done else "magenta",
        )

    def cli_event_handler(event: Event) -> None:
        title_suffix = f" · step {event.step}" if event.step is not None else ""
        debug_rows = []
        if debug:
            debug_rows = [
                (
                    "Event",
                    event.type.name if hasattr(event.type, "name") else str(event.type),
                ),
                ("Event ID", event.event_id),
                ("Run ID", event.run_id or "-"),
                ("Source", event.source or "-"),
            ]

        if _is_event_type(event, EVENT_AGENT_STARTED):
            rows = [
                ("Query", _short(event.payload.get("query"), 280)),
                ("Tools", _bullets(event.payload.get("tools", []))),
                *debug_rows,
            ]
            _panel("Agent started", "cyan", rows)
            return

        if _is_event_type(event, EVENT_INTENT_EXTRACTED):
            rows = [
                ("Intent", _short(event.payload.get("intent"), 260)),
                ("Goals", _bullets(event.payload.get("goals", []))),
                ("Constraints", _bullets(event.payload.get("constraints", []))),
                ("Open questions", _bullets(event.payload.get("open_questions", []))),
                *debug_rows,
            ]
            _panel(f"Intent extracted{title_suffix}", "blue", rows)
            return

        if _is_event_type(event, EVENT_STEP_STARTED):
            state = _state_payload(event)
            rows = [
                ("Status", state.get("status", "-")),
                ("Intent", _short(state.get("intent"), 220)),
                ("Goals", _bullets(state.get("goals", []), limit=5)),
                ("Open questions", _bullets(state.get("open_questions", []), limit=5)),
                (
                    "Facts",
                    len(state.get("working_memory", {}).get("facts", []))
                    if isinstance(state.get("working_memory", {}), dict)
                    else "-",
                ),
                *debug_rows,
            ]
            _panel(f"Step started{title_suffix}", "magenta", rows)
            return

        if _is_event_type(event, EVENT_CONTEXT_BUILT):
            rows = [
                ("Context length", event.payload.get("context_len", "-")),
                (
                    "Preview",
                    _short(
                        event.payload.get(
                            "context_preview", event.payload.get("context", "")
                        ),
                        360,
                    ),
                ),
                ("Tools", _bullets(event.payload.get("tools", []), limit=6)),
                (
                    "Artifacts",
                    event.payload.get(
                        "artifacts_count", event.payload.get("artifacts", "-")
                    ),
                ),
                *debug_rows,
            ]
            _panel(f"Context built{title_suffix}", "cyan", rows)
            return

        if _is_event_type(event, EVENT_THINKING_STARTED):
            state_status = event.payload.get("status", "planning")
            rows = [
                ("Status", state_status),
                *debug_rows,
            ]
            _panel(f"Thinking started{title_suffix}", "yellow", rows)
            return

        if _is_event_type(event, EVENT_REASONING_STARTED):
            role = event.payload.get("role", "model")
            key = (event.run_id, role)
            old = reasoning_live.pop(key, None)
            if old is not None:
                old.stop()  # defensive: a prior pass on this key never finished
            reasoning_text[key] = ""
            live = Live(
                _reasoning_panel(role, "", done=False),
                console=console,
                refresh_per_second=12,
            )
            live.start()
            reasoning_live[key] = live
            return

        if _is_event_type(event, EVENT_REASONING_CHUNK):
            role = event.payload.get("role", "model")
            key = (event.run_id, role)
            reasoning_text[key] = reasoning_text.get(key, "") + event.payload.get(
                "chunk", ""
            )
            live = reasoning_live.get(key)
            if live is not None:
                live.update(_reasoning_panel(role, reasoning_text[key], done=False))
            return

        if _is_event_type(event, EVENT_REASONING_FINISHED):
            role = event.payload.get("role", "model")
            key = (event.run_id, role)
            full = event.payload.get("reasoning", reasoning_text.get(key, ""))
            live = reasoning_live.pop(key, None)
            reasoning_text.pop(key, None)
            if live is not None:
                live.update(_reasoning_panel(role, full, done=True))
                live.stop()
            else:
                # No STARTED/CHUNK events were seen for this key (e.g. handler
                # subscribed mid-run) — fall back to a plain panel.
                _panel(
                    f"{role.capitalize()} reasoning{title_suffix}",
                    "magenta",
                    [("Reasoning", _short(full, 600)), *debug_rows],
                )
            return

        if _is_event_type(event, EVENT_THINKING_ENDED):
            action = _action_payload(event)
            rows = [
                ("Tool", action.get("tool", "-")),
                ("Input", _short(action.get("input"), 220)),
                ("Reason", _short(action.get("reason"), 220)),
                ("Duration", _duration_text(event.duration_ms)),
                *debug_rows,
            ]
            _panel(f"Thinking ended{title_suffix}", "yellow", rows)
            return

        if _is_event_type(event, EVENT_ACTION_PLANNED):
            action = _action_payload(event)
            rows = [
                ("Tool", action.get("tool", "-")),
                ("Input", _short(action.get("input"), 220)),
                ("Reason", _short(action.get("reason"), 220)),
                ("Done", action.get("done", False)),
                *debug_rows,
            ]
            _panel(f"Action planned{title_suffix}", "yellow", rows)
            return

        if _is_event_type(event, EVENT_ACTION_STARTED, EVENT_ACTION_EXECUTING):
            action = _action_payload(event)
            rows = [
                ("Tool", action.get("tool", event.payload.get("tool", "-"))),
                (
                    "Input",
                    _short(action.get("input", event.payload.get("input", "")), 220),
                ),
                ("Status", "running"),
                *debug_rows,
            ]
            _panel(f"Action started{title_suffix}", "blue", rows)
            return

        if _is_event_type(event, EVENT_ACTION_FINISHED, EVENT_ACTION_EXECUTED):
            result = _result_payload(event)
            success = bool(event.payload.get("success", result.get("success", False)))
            summary = event.payload.get("summary", result.get("summary", ""))
            artifacts = result.get("artifacts", event.payload.get("artifacts", []))
            rows = [
                ("Result", "success" if success else "failed"),
                ("Summary", _short(summary, 280)),
                ("Data", _short(result.get("data"), 280)),
                ("Artifacts", _bullets(artifacts, limit=5)),
                ("Duration", _duration_text(event.duration_ms)),
                *debug_rows,
            ]
            _panel(
                f"Action finished{title_suffix}", "green" if success else "red", rows
            )
            return

        if _is_event_type(event, EVENT_VALIDATION_STARTED):
            rows = [
                ("Status", "validating"),
                *debug_rows,
            ]
            _panel(f"Validation started{title_suffix}", "dim", rows)
            return

        if _is_event_type(event, EVENT_VALIDATION_FINISHED, EVENT_VALIDATION_DONE):
            validation = _validation_payload(event)
            status = str(event.payload.get("status", validation.get("status", "-")))
            missing = event.payload.get("missing", validation.get("missing", []))
            rows = [
                ("Status", status),
                (
                    "Reason",
                    _short(
                        event.payload.get("reason", validation.get("reason", "")), 260
                    ),
                ),
                ("Missing", _bullets(missing, limit=6)),
                ("Duration", _duration_text(event.duration_ms)),
                *debug_rows,
            ]
            _panel(
                f"Validation finished{title_suffix}", _validation_color(status), rows
            )
            return

        if _is_event_type(event, EVENT_SYNTHESIS_STARTED):
            rows = [
                ("Status", event.payload.get("status", "synthesizing")),
                *debug_rows,
            ]
            _panel(f"Synthesis started{title_suffix}", "cyan", rows)
            return

        if _is_event_type(event, EVENT_SYNTHESIS_FINISHED):
            answer = _answer_payload(event)
            rows = [
                (
                    "Output",
                    _short(answer.get("output", answer.get("response", "")), 320),
                ),
                ("References", _bullets(answer.get("references", []), limit=5)),
                (
                    "Output length",
                    event.payload.get(
                        "output_len", len(_fmt(answer.get("output", "")))
                    ),
                ),
                *debug_rows,
            ]
            _panel(f"Synthesis finished{title_suffix}", "green", rows)
            return

        if _is_event_type(event, EVENT_STATE_UPDATED):
            state = _state_payload(event)
            rows = [
                ("Status", state.get("status", "-")),
                ("Note", _short(event.payload.get("note", ""), 260)),
                (
                    "Step",
                    state.get(
                        "step_index", event.step if event.step is not None else "-"
                    ),
                ),
                (
                    "Errors",
                    len(state.get("working_memory", {}).get("errors", []))
                    if isinstance(state.get("working_memory", {}), dict)
                    else "-",
                ),
                *debug_rows,
            ]
            _panel(f"State updated{title_suffix}", "white", rows)
            return

        if _is_event_type(event, EVENT_MEMORY_UPDATED):
            rows = [
                ("Key", event.payload.get("key", "-")),
                ("Value", _short(event.payload.get("value", ""), 280)),
                *debug_rows,
            ]
            _panel(f"Memory updated{title_suffix}", "cyan", rows)
            return

        if _is_event_type(event, EVENT_TOOL_ERROR):
            rows = [
                ("Tool", event.payload.get("tool", "-")),
                ("Error", _short(event.error, 260)),
                ("Traceback", _short(event.payload.get("traceback", ""), 320)),
                *debug_rows,
            ]
            _panel(f"Tool error{title_suffix}", "red", rows)
            return

        if _is_event_type(event, EVENT_WARNING):
            rows = [
                ("Message", _short(event.payload.get("message", ""), 260)),
                ("Listener", event.payload.get("listener", "-")),
                ("Error", _short(event.error, 260)),
                *debug_rows,
            ]
            _panel("Warning", "yellow", rows)
            return

        if _is_event_type(event, EVENT_AGENT_DONE):
            state = _state_payload(event)
            output = _answer_payload(event).get(
                "output", event.payload.get("output", None)
            )
            working_memory = (
                state.get("working_memory", {}) if isinstance(state, dict) else {}
            )
            facts = (
                working_memory.get("facts", [])
                if isinstance(working_memory, dict)
                else []
            )
            results = (
                working_memory.get("results", [])
                if isinstance(working_memory, dict)
                else []
            )
            errors = (
                working_memory.get("errors", [])
                if isinstance(working_memory, dict)
                else []
            )
            artifacts = state.get("artifacts", []) if isinstance(state, dict) else []

            rows = [
                (
                    "Status",
                    state.get("status", "done") if isinstance(state, dict) else "done",
                ),
                (
                    "Steps",
                    state.get("step_index", "?") if isinstance(state, dict) else "?",
                ),
                ("Facts", len(facts) if isinstance(facts, list) else "-"),
                ("Results", len(results) if isinstance(results, list) else "-"),
                ("Errors", len(errors) if isinstance(errors, list) else "-"),
                ("Artifacts", len(artifacts) if isinstance(artifacts, list) else "-"),
                ("Output", _short(output, 320)),
                *debug_rows,
            ]
            _panel("Agent finished", "green", rows)
            console.print("[bold green]✓ Agent finished[/bold green]")
            return

        if _is_event_type(event, EVENT_AGENT_ERROR):
            rows = [
                ("Error", _short(event.error, 280)),
                ("Query", _short(event.payload.get("query", ""), 260)),
                ("Traceback", _short(event.payload.get("traceback", ""), 360)),
                ("Step", event.step if event.step is not None else "-"),
                *debug_rows,
            ]
            _panel("Agent error", "red", rows)
            console.print(f"[bold red]✗ {event.error}[/bold red]")
            return

        # Fallback for unhandled events in debug mode.
        if debug:
            _panel(
                "Event",
                "white",
                [
                    (
                        "Type",
                        event.type.name
                        if hasattr(event.type, "name")
                        else str(event.type),
                    ),
                    ("Payload", _short(event.payload, 360)),
                    ("Duration", _duration_text(event.duration_ms)),
                    ("Run ID", event.run_id or "-"),
                    ("Event ID", event.event_id),
                ],
            )

    return cli_event_handler


def confirm(message: str, default: bool = False) -> bool:
    return bool(questionary.confirm(message, default=default).ask())


def _choose_installed_model() -> str:
    models = list_models()

    if not models:
        console.print("[yellow]No local GGUF models found.[/yellow]")
        raise typer.Exit()

    choices: list[str] = []
    for model in models:
        choices.extend(model.files)

    selected = questionary.select(
        "Which model should be used?",
        choices=choices,
    ).ask()

    if selected is None:
        raise typer.Exit()

    return selected


def _print_run_summary(state: Any) -> None:
    console.rule("[bold green]Result")

    snapshot = getattr(state, "state_snapshot", None)
    if callable(snapshot):
        console.print(Panel(snapshot(), title="Final state", border_style="green"))  # pyright: ignore[reportArgumentType]

    working_memory = getattr(state, "working_memory", {})
    if isinstance(working_memory, dict) and working_memory:
        table = Table(box=box.SIMPLE, show_header=True, expand=True)
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", overflow="fold")
        for key, value in working_memory.items():
            if isinstance(value, list):
                preview = _bullets(value, limit=8)
            else:
                preview = _short(value, 320)
            table.add_row(str(key), preview)
        console.print(Panel(table, title="Working memory", border_style="cyan"))

    step_log = getattr(state, "step_log", [])
    if step_log:
        console.print(
            Panel(
                _bullets(step_log, limit=10), title="Step log", border_style="magenta"
            )
        )

    artifacts = getattr(state, "artifacts", [])
    if artifacts:
        table = Table(box=box.SIMPLE, show_header=True, expand=True)
        table.add_column("Name", style="bold")
        table.add_column("Type")
        table.add_column("Tags")
        table.add_column("Description", overflow="fold")
        for art in artifacts[-12:]:
            name = getattr(art, "name", "-")
            typ = getattr(art, "type", "-")
            tags = (
                ",".join(getattr(art, "tags", [])[:4])
                if getattr(art, "tags", None)
                else "-"
            )
            desc = _short(getattr(art, "description", "-"), 180)
            table.add_row(str(name), str(typ), tags, desc)
        console.print(Panel(table, title="Artifacts", border_style="yellow"))


async def _run(query: str, model: str, *, debug: bool = False):
    try:
        model_path = resolve_model_path(model)

        if not model_path:
            console.print(f"[red]Model '{model}' not found.[/red]")
            raise typer.Exit(1)

        conf = AgentConfig.from_path(str(model_path))
        agent = Agent(conf)

        agent.event_bus.subscribe(build_cli_event_handler(debug=debug))

        console.print(f"[green]Running:[/green] {model_path.name}")
        console.print(f"[dim]Query:[/dim] {query}")

        state = await agent.run(query)

        _print_run_summary(state)

    except typer.Exit:
        raise
    except Exception as e:
        arc_status("error", str(e))
        console.print_exception()
        raise typer.Exit(1)


def run(
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model name or path to use."
    ),
    query: str | None = typer.Option(
        None, "--query", "-q", help="Query to send to the agent."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show extra event metadata and unhandled events."
    ),
):
    if not model:
        model = _choose_installed_model()

    if not query:
        query = questionary.text("What would you like to ask?").ask()
        if not query:
            raise typer.Exit(1)

    asyncio.run(_run(query, model, debug=debug))