from __future__ import annotations

import time
import uuid
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from src.memory import AgentState
from src.tools.registry import ToolResult
from src.roles import ValidationResult


class EventType(Enum):
    AGENT_STARTED = auto()
    INTENT_EXTRACTED = auto()
    STEP_STARTED = auto()
    CONTEXT_BUILT = auto()
    THINKING_STARTED = auto()
    THINKING_ENDED = auto()
    ACTION_PLANNED = auto()
    ACTION_EXECUTING = auto()
    ACTION_EXECUTED = auto()
    VALIDATION_STARTED = auto()
    VALIDATION_DONE = auto()
    MEMORY_UPDATED = auto()
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
        self._listeners = [l for l in self._listeners if l is not fn]

    def emit(self, event: Event) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        for fn in tuple(self._listeners):
            try:
                fn(event)
            except Exception as exc:
                # Never break agent execution because of a listener
                err = Event(
                    type=EventType.WARNING,
                    payload={"listener": getattr(fn, "__name__", str(fn))},
                    error=f"{type(exc).__name__}: {exc}",
                    source="event_bus",
                )
                self._history.append(err)

    def history(self) -> list[Event]:
        return list(self._history)


def _get_state(**kwargs) -> AgentState:
    state = kwargs.get("state")
    if not isinstance(state, AgentState):
        raise ValueError("Kwarg key state is not of type AgentState!")

    return state


def _get_t(tn: str, **kwargs) -> float:
    tnv = kwargs.get(tn)

    if not isinstance(tnv, float):
        raise ValueError(f"Kwargs key {tn} of value {tnv} must be float.")

    return tnv  # type: ignore


def emit(event_bus: EventBus, id: int, **kwargs):
    """
    Emit the an agent-event by id and required kwargs.

    :param event_bus: The EventBus instance
    :param id: the id of the event

    Event Ids and their Kwargs:

    > ! every event needs the 'run_id' !

    - 0 -> query
    - 1 -> state
    - 2 -> state
    - 3 -> state, context
    - 4 -> state
    - 5 -> state, action, t0
    - 6 -> state, action
    - 7 -> state, result, t1
    - 8 -> state
    - 9 -> state, validation, t2
    - 10 ->  state
    - 11 -> state, query, exc

    """

    if not kwargs:
        raise ValueError("Missing kwargs!")

    run_id = kwargs.get("run_id")

    if not run_id:
        raise ValueError("Missing or Invalid param 'run_id' for kwargs!")

    if id == 0:
        event_bus.emit(
            Event(
                EventType.AGENT_STARTED,
                {"query": kwargs.get("query")},
                run_id=run_id,
            )
        )
        return

    state = _get_state(**kwargs)

    match id:
        case 1:
            event_bus.emit(
                Event(
                    EventType.INTENT_EXTRACTED,
                    {"intent": state.intent, "goals": state.goals},
                    run_id=run_id,
                )
            )

        case 2:
            event_bus.emit(
                Event(
                    EventType.STEP_STARTED,
                    {"state": state.as_dict()},
                    run_id=run_id,
                    step=state.step_index,
                )
            )

        case 3:
            event_bus.emit(
                Event(
                    EventType.CONTEXT_BUILT,
                    {
                        "context_len": len(str(kwargs.get("context"))),
                        "tools": kwargs.get("tools"),
                        "base": kwargs.get("base"),
                        "artifacts": kwargs.get("artifacts"),
                    },
                    run_id=run_id,
                    step=state.step_index,
                )
            )

        case 4:
            event_bus.emit(
                Event(
                    EventType.THINKING_STARTED,
                    {},
                    run_id=run_id,
                    step=state.step_index,
                )
            )

        case 5:
            event_bus.emit(
                Event(
                    EventType.THINKING_ENDED,
                    {"action": kwargs.get("action")},
                    run_id=run_id,
                    step=state.step_index,
                    duration_ms=(time.time() - _get_t("t0", **kwargs)) * 1000,
                )
            )
        case 6:
            event_bus.emit(
                Event(
                    EventType.ACTION_PLANNED,
                    {"action": kwargs.get("action")},
                    run_id=run_id,
                    step=state.step_index,
                )
            )
        case 7:
            result = kwargs.get("result")
            if not isinstance(result, ToolResult):
                raise ValueError("Kwargs key 'result' must be type ToolResult!")
            event_bus.emit(
                Event(
                    EventType.ACTION_EXECUTED,
                    {"success": result.success, "summary": result.summary},
                    run_id=run_id,
                    step=state.step_index,
                    duration_ms=(time.time() - _get_t("t1", **kwargs)) * 1000,
                )
            )
        case 8:
            event_bus.emit(
                Event(
                    EventType.VALIDATION_STARTED,
                    {},
                    run_id=run_id,
                    step=state.step_index,
                )
            )
        case 9:
            validation = kwargs.get("validation")
            if not isinstance(validation, ValidationResult):
                raise ValueError("Kwargs key 'result' must be type ToolResult!")
            event_bus.emit(
                Event(
                    EventType.VALIDATION_DONE,
                    {"status": validation.status, "reason": validation.reason},
                    run_id=run_id,
                    step=state.step_index,
                    duration_ms=(time.time() - _get_t("t2", **kwargs)) * 1000,
                )
            )
        case 10:
            event_bus.emit(
                Event(EventType.AGENT_DONE, {"state": state.as_dict()}, run_id=run_id)
            )
        case 11:
            exc = kwargs.get("exc")
            event_bus.emit(
                Event(
                    EventType.AGENT_ERROR,
                    {"query": kwargs.get("query"), "traceback": traceback.format_exc()},
                    run_id=run_id,
                    step=getattr(state, "step_index", None),  # type: ignore
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
