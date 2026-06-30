from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from src.memory import AgentState
from src.roles import ValidationResult
from src.schema import ToolResult


class EventType(Enum):
    AGENT_STARTED = auto()
    INTENT_EXTRACTED = auto()
    STEP_STARTED = auto()
    CONTEXT_BUILT = auto()
    THINKING_STARTED = auto()
    ACTION_PLANNED = auto()
    ACTION_STARTED = auto()
    ACTION_FINISHED = auto()
    VALIDATION_STARTED = auto()
    VALIDATION_FINISHED = auto()
    SYNTHESIS_STARTED = auto()
    SYNTHESIS_FINISHED = auto()
    STATE_UPDATED = auto()
    AGENT_DONE = auto()
    AGENT_ERROR = auto()
    TOOL_ERROR = auto()
    WARNING = auto()


@dataclass(slots=True)
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    run_id: str = ""
    step: int | None = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str = "agent"
    duration_ms: float | None = None
    error: str | None = None


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Callable[[Event], None]] = []
        self._history: list[Event] = []
        self._max_history = 500

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        if fn not in self._listeners:
            self._listeners.append(fn)

    def unsubscribe(self, fn: Callable[[Event], None]) -> None:
        self._listeners = [
            listener for listener in self._listeners if listener is not fn
        ]

    def emit(self, event: Event) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        for fn in tuple(self._listeners):
            try:
                fn(event)
            except Exception as exc:
                self._history.append(
                    Event(
                        type=EventType.WARNING,
                        payload={
                            "listener": getattr(fn, "__name__", str(fn)),
                            "message": "listener raised while handling event",
                        },
                        source="event_bus",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    def history(self) -> list[Event]:
        return list(self._history)


def _ensure_state(kwargs: dict[str, Any]) -> AgentState:
    state = kwargs.get("state")
    if not isinstance(state, AgentState):
        raise ValueError("Missing or invalid 'state' kwarg; expected AgentState.")
    return state


def _ensure_run_id(kwargs: dict[str, Any]) -> str:
    run_id = kwargs.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Missing or invalid 'run_id' kwarg.")
    return run_id


def _duration_ms(start: Any) -> float:
    if not isinstance(start, (float, int)):
        raise ValueError(f"Expected numeric start time, got {type(start).__name__}.")
    return (time.time() - float(start)) * 1000.0


def _validate_tool_result(result: Any) -> ToolResult:
    if not isinstance(result, ToolResult):
        raise ValueError("Kwarg 'result' must be a ToolResult instance.")
    return result


def _validate_validation_result(result: Any) -> ValidationResult:
    if not isinstance(result, ValidationResult):
        raise ValueError("Kwarg 'validation' must be a ValidationResult instance.")
    return result


def emit_event(
    event_bus: EventBus,
    event_type: EventType,
    *,
    run_id: str,
    step: int | None = None,
    payload: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
    source: str = "agent",
) -> None:
    event_bus.emit(
        Event(
            type=event_type,
            payload=payload or {},
            run_id=run_id,
            step=step,
            duration_ms=duration_ms,
            error=error,
            source=source,
        )
    )


def emit_agent_started(
    event_bus: EventBus, *, run_id: str, query: str, tools: list[str] | None = None
) -> None:
    emit_event(
        event_bus,
        EventType.AGENT_STARTED,
        run_id=run_id,
        payload={
            "query": query,
            "tools": tools or [],
        },
    )


def emit_intent_extracted(
    event_bus: EventBus, *, run_id: str, state: AgentState
) -> None:
    emit_event(
        event_bus,
        EventType.INTENT_EXTRACTED,
        run_id=run_id,
        step=state.step_index,
        payload={
            "intent": state.intent,
            "goals": list(state.goals),
            "constraints": list(state.constraints),
            "open_questions": list(state.open_questions),
        },
    )


def emit_step_started(event_bus: EventBus, *, run_id: str, state: AgentState) -> None:
    emit_event(
        event_bus,
        EventType.STEP_STARTED,
        run_id=run_id,
        step=state.step_index,
        payload={
            "state": state.to_dict(),
            "status": state.status,
        },
    )


def emit_context_built(
    event_bus: EventBus,
    *,
    run_id: str,
    state: AgentState,
    context: str,
    tools: list[str] | None = None,
) -> None:
    emit_event(
        event_bus,
        EventType.CONTEXT_BUILT,
        run_id=run_id,
        step=state.step_index,
        payload={
            "context_len": len(context),
            "context_preview": context[:500],
            "tools": tools or [],
            "artifacts_count": len(state.artifacts),
        },
    )


def emit_thinking_started(
    event_bus: EventBus, *, run_id: str, state: AgentState
) -> None:
    emit_event(
        event_bus,
        EventType.THINKING_STARTED,
        run_id=run_id,
        step=state.step_index,
        payload={"status": state.status},
    )


def emit_action_planned(
    event_bus: EventBus, *, run_id: str, state: AgentState, action: Any
) -> None:
    emit_event(
        event_bus,
        EventType.ACTION_PLANNED,
        run_id=run_id,
        step=state.step_index,
        payload={
            "action": action,
            "tool": getattr(action, "tool", None),
            "input": getattr(action, "input", None),
            "reason": getattr(action, "reason", None),
        },
    )


def emit_action_started(
    event_bus: EventBus, *, run_id: str, state: AgentState, action: Any
) -> None:
    emit_event(
        event_bus,
        EventType.ACTION_STARTED,
        run_id=run_id,
        step=state.step_index,
        payload={
            "tool": getattr(action, "tool", None),
            "input": getattr(action, "input", None),
        },
    )


def emit_action_finished(
    event_bus: EventBus,
    *,
    run_id: str,
    state: AgentState,
    result: ToolResult,
    t0: Any,
) -> None:
    emit_event(
        event_bus,
        EventType.ACTION_FINISHED,
        run_id=run_id,
        step=state.step_index,
        duration_ms=_duration_ms(t0),
        payload={
            "success": result.success,
            "summary": result.summary,
            "data": getattr(result, "data", None),
            "artifacts": [a.name for a in getattr(result, "artifacts", []) or []],
        },
    )


def emit_validation_started(
    event_bus: EventBus, *, run_id: str, state: AgentState
) -> None:
    emit_event(
        event_bus,
        EventType.VALIDATION_STARTED,
        run_id=run_id,
        step=state.step_index,
        payload={"status": state.status},
    )


def emit_validation_finished(
    event_bus: EventBus,
    *,
    run_id: str,
    state: AgentState,
    validation: ValidationResult,
    t2: Any,
) -> None:
    emit_event(
        event_bus,
        EventType.VALIDATION_FINISHED,
        run_id=run_id,
        step=state.step_index,
        duration_ms=_duration_ms(t2),
        payload={
            "status": validation.status,
            "reason": validation.reason,
            "missing": list(getattr(validation, "missing", []) or []),
        },
    )


def emit_synthesis_started(
    event_bus: EventBus, *, run_id: str, state: AgentState
) -> None:
    emit_event(
        event_bus,
        EventType.SYNTHESIS_STARTED,
        run_id=run_id,
        step=state.step_index,
        payload={"status": state.status},
    )


def emit_synthesis_finished(
    event_bus: EventBus,
    *,
    run_id: str,
    state: AgentState,
    output: str,
) -> None:
    emit_event(
        event_bus,
        EventType.SYNTHESIS_FINISHED,
        run_id=run_id,
        step=state.step_index,
        payload={
            "output": output,
            "output_len": len(output),
        },
    )


def emit_state_updated(
    event_bus: EventBus,
    *,
    run_id: str,
    state: AgentState,
    note: str | None = None,
) -> None:
    emit_event(
        event_bus,
        EventType.STATE_UPDATED,
        run_id=run_id,
        step=state.step_index,
        payload={
            "state": state.to_dict(),
            "note": note,
            "status": state.status,
        },
    )


def emit_agent_done(
    event_bus: EventBus,
    *,
    run_id: str,
    state: AgentState,
    output: str | None = None,
) -> None:
    emit_event(
        event_bus,
        EventType.AGENT_DONE,
        run_id=run_id,
        step=state.step_index,
        payload={
            "state": state.to_dict(),
            "output": output,
        },
    )


def emit_tool_error(
    event_bus: EventBus,
    *,
    run_id: str,
    state: AgentState,
    tool_name: str,
    exc: Exception,
) -> None:
    emit_event(
        event_bus,
        EventType.TOOL_ERROR,
        run_id=run_id,
        step=state.step_index,
        error=f"{type(exc).__name__}: {exc}",
        payload={
            "tool": tool_name,
            "traceback": traceback.format_exc(),
        },
    )


def emit_agent_error(
    event_bus: EventBus,
    *,
    run_id: str,
    state: AgentState | None,
    query: str,
    exc: Exception,
) -> None:
    emit_event(
        event_bus,
        EventType.AGENT_ERROR,
        run_id=run_id,
        step=getattr(state, "step_index", None),
        error=f"{type(exc).__name__}: {exc}",
        payload={
            "query": query,
            "state": state.to_dict() if state is not None else None,
            "traceback": traceback.format_exc(),
        },
    )


_EVENT_ID_MAP: dict[int, EventType] = {
    0: EventType.AGENT_STARTED,
    1: EventType.INTENT_EXTRACTED,
    2: EventType.STEP_STARTED,
    3: EventType.CONTEXT_BUILT,
    4: EventType.THINKING_STARTED,
    5: EventType.ACTION_PLANNED,
    6: EventType.ACTION_STARTED,
    7: EventType.ACTION_FINISHED,
    8: EventType.VALIDATION_STARTED,
    9: EventType.VALIDATION_FINISHED,
    10: EventType.AGENT_DONE,
    11: EventType.AGENT_ERROR,
}


def emit(event_bus: EventBus, id: int, **kwargs) -> None:
    """
    Backwards-compatible event emission wrapper.

    Prefer the typed helpers above for new code.
    """

    if not kwargs:
        raise ValueError("Missing kwargs!")

    run_id = _ensure_run_id(kwargs)

    if id == 0:
        emit_agent_started(
            event_bus,
            run_id=run_id,
            query=str(kwargs.get("query", "")),
            tools=list(kwargs.get("tools") or []),
        )
        return

    state = _ensure_state(kwargs)

    match id:
        case 1:
            emit_intent_extracted(event_bus, run_id=run_id, state=state)
        case 2:
            emit_step_started(event_bus, run_id=run_id, state=state)
        case 3:
            emit_context_built(
                event_bus,
                run_id=run_id,
                state=state,
                context=str(kwargs.get("context", "")),
                tools=list(kwargs.get("tools") or []),
            )
        case 4:
            emit_thinking_started(event_bus, run_id=run_id, state=state)
        case 5:
            emit_action_planned(
                event_bus,
                run_id=run_id,
                state=state,
                action=kwargs.get("action"),
            )
        case 6:
            emit_action_started(
                event_bus,
                run_id=run_id,
                state=state,
                action=kwargs.get("action"),
            )
        case 7:
            result = _validate_tool_result(kwargs.get("result"))
            emit_action_finished(
                event_bus,
                run_id=run_id,
                state=state,
                result=result,
                t0=kwargs.get("t1"),
            )
        case 8:
            emit_validation_started(event_bus, run_id=run_id, state=state)
        case 9:
            validation = _validate_validation_result(kwargs.get("validation"))
            emit_validation_finished(
                event_bus,
                run_id=run_id,
                state=state,
                validation=validation,
                t2=kwargs.get("t2"),
            )
        case 10:
            emit_agent_done(
                event_bus,
                run_id=run_id,
                state=state,
                output=str(kwargs.get("output"))
                if kwargs.get("output") is not None
                else None,
            )
        case 11:
            exc = kwargs.get("exc")
            if not isinstance(exc, Exception):
                raise ValueError("Kwarg 'exc' must be an Exception instance.")
            emit_agent_error(
                event_bus,
                run_id=run_id,
                state=state,
                query=str(kwargs.get("query", "")),
                exc=exc,
            )
        case _:
            raise ValueError(f"Unknown event id: {id}")
