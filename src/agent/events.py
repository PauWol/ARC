from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from src.agent.memory import AgentState
from src.agent.roles import ValidationResult
from src.agent.schema import ToolResult


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
    REASONING_STARTED = auto()
    REASONING_CHUNK = auto()
    REASONING_FINISHED = auto()


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


class EventEmitter(EventBus):
    def __init__(self, id: str) -> None:
        super().__init__()
        self.id = id

    def agent_started(self):
        ev = Event(EventType.AGENT_STARTED)
