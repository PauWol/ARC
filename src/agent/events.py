from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar

from src.agent.memory import AgentState, Session, WorkingMemory
from src.agent.roles import ValidationResult
from src.agent.schema import ToolResult


class EventType(Enum):
    AGENT_STARTED = auto()
    INTENT_EXTRACTED = auto()
    STEP_STARTED = auto()
    THINKING_STARTED = auto()
    THINKING_FINISHED = auto()
    ACTION_STARTED = auto()
    ACTION_FINISHED = auto()
    VALIDATION_STARTED = auto()
    VALIDATION_FINISHED = auto()
    SYNTHESIS_STARTED = auto()
    SYNTHESIS_FINISHED = auto()
    STATE_UPDATED = auto()
    AGENT_DONE = auto()
    AGENT_ERROR = auto()
    REASONING_STARTED = auto()
    REASONING_CHUNK = auto()
    REASONING_FINISHED = auto()


@dataclass(slots=True, frozen=True)
class AgentStarted:
    run_id: str


@dataclass(slots=True, frozen=True)
class IntentExtracted:
    intent: str


@dataclass(slots=True, frozen=True)
class StepStarted:
    step: int
    description: str


@dataclass(slots=True, frozen=True)
class ThinkingStarted:
    pass


@dataclass(slots=True, frozen=True)
class ThinkingFinished:
    tool: str
    reason: str
    input: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ActionStarted:
    step: int
    tool: str


@dataclass(slots=True, frozen=True)
class ActionFinished:
    step: int
    result: ToolResult
    duration_ms: float


@dataclass(slots=True, frozen=True)
class ValidationStarted:
    validation: ValidationResult


@dataclass(slots=True, frozen=True)
class ValidationFinished:
    validation: ValidationResult


@dataclass(slots=True, frozen=True)
class AgentDone:
    answer: str


@dataclass(slots=True, frozen=True)
class AgentError:
    error: str


EVENT_TYPES: dict[type, EventType] = {
    AgentStarted: EventType.AGENT_STARTED,
    IntentExtracted: EventType.INTENT_EXTRACTED,
    StepStarted: EventType.STEP_STARTED,
    ThinkingStarted: EventType.THINKING_STARTED,
    ThinkingFinished: EventType.THINKING_FINISHED,
    ActionStarted: EventType.ACTION_STARTED,
    ActionFinished: EventType.ACTION_FINISHED,
    ValidationFinished: EventType.VALIDATION_FINISHED,
    AgentDone: EventType.AGENT_DONE,
    AgentError: EventType.AGENT_ERROR,
}

P = TypeVar("P")


@dataclass(slots=True)
class Event(Generic[P]):
    payload: P

    run_id: str

    type: EventType = field(init=False)

    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    source: str = "agent"

    def __post_init__(self):
        self.type = EVENT_TYPES[type(self.payload)]


Listener = Callable[[Event], None]


class EventBus:
    def __init__(self):
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener):
        self._listeners.append(listener)

    def unsubscribe(self, listener: Listener):
        self._listeners.remove(listener)

    def emit(self, event: Event):
        for listener in tuple(self._listeners):
            listener(event)


class EventEmitter(EventBus):
    def __init__(self, run_id: str):
        self._session_id = run_id

    def emit_(self, payload: P) -> None:
        self.emit(
            Event(
                payload=payload,
                run_id=self._session_id,
            )
        )

    @classmethod
    def with_agent_started(cls, session_id: str):
        _cls = cls(session_id)
        _cls.agent_started()
        return _cls

    # Convenience wrappers

    def agent_started(self):
        self.emit_(AgentStarted(run_id=self._session_id))

    def thinking_started(self):
        self.emit_(ThinkingStarted())

    def thinking_finished(self, tool: str, reason: str, input: dict[str, Any]):
        self.emit_(ThinkingFinished(tool, reason, input))

    def step_started(self, step: int, description: str):
        self.emit_(StepStarted(step, description))

    def action_started(self, step: int, tool: str):
        self.emit_(ActionStarted(step, tool))

    def action_finished(
        self,
        step: int,
        result: ToolResult,
        duration_ms: float,
    ):
        self.emit_(ActionFinished(step, result, duration_ms))

    def validation_finished(self, validation: ValidationResult):
        self.emit_(ValidationFinished(validation))

    def done(self, answer: str):
        self.emit_(AgentDone(answer))

    def error(self, error: str):
        self.emit_(AgentError(error))
