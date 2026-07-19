from __future__ import annotations

import re
from dataclasses import dataclass, field
import time
from typing import Any, Literal
import uuid

from src.agent.events import EventBus
from src.agent.schema import Artifact

STATE_VALUES = {
    "new",
    "planned",
    "working",
    "present",
    "done",
    "replan",
    "error",
}
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]{3,}`")
_URL_RE = re.compile(r"https?://[^\s'\"<>()\[\]]+|www\.[^\s'\"<>()\[\]]+", re.I)
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\[\w\\.\-/]+)"
    r"|(?:/[\w.\-/]+(?:\.\w+)?)"
    r"|(?:\.{1,2}/[\w.\-/]+(?:\.\w+)?)",
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_context_noise(text: str) -> str:
    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _PATH_RE.sub(" ", text)
    return normalize_text(text)


def extract_context_items(text: str) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()

    def add(tag: str, value: str) -> None:
        value = normalize_text(value)
        if not value:
            return
        key = f"{tag}:{value.lower()}"
        if key in seen:
            return
        seen.add(key)
        items.append(f"{tag}: {value}")

    for m in _CODE_BLOCK_RE.finditer(text):
        add("code", m.group().strip("`"))

    for m in _INLINE_CODE_RE.finditer(text):
        add("code", m.group().strip("`"))

    for m in _URL_RE.finditer(text):
        add("url", m.group())

    for m in _PATH_RE.finditer(text):
        add("path", m.group())

    return items


@dataclass(slots=True)
class AgentState:
    """
    Working memory for a single agent task.
    """

    raw_input: str
    cleaned_input: str = ""

    intent: str = ""
    goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    context_items: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    working_memory: dict[str, list[str]] = field(
        default_factory=lambda: {
            "facts": [],
            "results": [],
            "errors": [],
            "temp": [],
        }
    )
    step_log: list[str] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)

    step_index: int = 0
    replan_index: int = 0
    status: str = "new"
    error: str | None = None

    @classmethod
    def from_user_input(cls, text: str) -> "AgentState":
        return cls(
            raw_input=text,
            cleaned_input=strip_context_noise(text),
            context_items=extract_context_items(text),
        )

    def remember(self, key: str, value: Any) -> None:
        self.working_memory[key] = value

    def register_artifact(self, art: Artifact):
        self.artifacts.append(art)

    def extend_artifacts(self, arts: list[Artifact]):
        self.artifacts.extend(arts)

    def remember_fact(self, value: str) -> None:
        self.working_memory["facts"].append(value)

    def remember_error(self, value: str) -> None:
        self.working_memory["errors"].append(value)

    def remember_result(self, value: str):
        self.working_memory["results"].append(value)

    def set_status(self, new_status: str) -> None:
        if new_status not in STATE_VALUES:
            raise ValueError(
                f"Transition to state {new_status} not valid. State not supported!"
            )

        self.status = new_status

    def forget(self, key: str) -> None:
        self.working_memory.pop(key, None)

    def advance(self, note: str | None = None) -> None:
        self.step_index += 1

        entry = f"step {self.step_index}"
        if note:
            entry += f": {note}"

        self.step_log.append(entry)

        self.remember(f"step_{self.step_index}", note or "")

    def _format_list(self, title: str, items: list[str]) -> list[str]:
        if not items:
            return []
        return [f"{title}:"] + [f"- {i}" for i in items]

    def _artifact_summary(self, a: Artifact) -> str:
        tags = ",".join(a.tags[:4]) if a.tags else ""
        return f"{a.type} | {a.name} | {tags} | {a.description}"

    def compact_prompt(self, extra_artifacts: list[Artifact] | None = None) -> str:
        extra_artifacts = extra_artifacts or []

        lines: list[str] = []

        if self.intent:
            lines += ["INTENT:", self.intent]

        lines += self._format_list("GOALS", self.goals)
        lines += self._format_list("CONSTRAINTS", self.constraints)
        lines += self._format_list("CONTEXT", self.context_items)
        lines += self._format_list("OPEN_QUESTIONS", self.open_questions)

        if self.working_memory:
            lines.append("WORKING_MEMORY:")
            for k, v in self.working_memory.items():
                lines.append(f"- {k}: {v}")

        if extra_artifacts:
            lines.append("ARTIFACTS:")
            for a in extra_artifacts:
                lines.append(f"- {self._artifact_summary(a)}")

        return "\n".join(lines)

    def state_snapshot(self) -> str:
        return "\n".join(
            [
                f"STATUS: {self.status}",
                f"STEP: {self.step_index}",
                f"INTENT: {self.intent}",
                f"FACTS: {self.working_memory.get('facts', [])[:5]}",
                f"ERRORS: {self.working_memory.get('errors', [])[-3:]}",
            ]
        )

    @property
    def is_done(self) -> bool:
        return self.status == "done"

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_input": self.raw_input,
            "cleaned_input": self.cleaned_input,
            "intent": self.intent,
            "goals": list(self.goals),
            "constraints": list(self.constraints),
            "context_items": list(self.context_items),
            "open_questions": list(self.open_questions),
            "working_memory": dict(self.working_memory),
            "step_index": self.step_index,
            "status": self.status,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"AgentState(intent={self.intent!r}, goals={self.goals}, "
            f"constraints={self.constraints}, step={self.step_index}, "
            f"status={self.status!r})"
        )


def build_initial_state(query: str) -> AgentState:
    return AgentState.from_user_input(query)


@dataclass
class Session:
    query: str
    id: str
    event_bus: EventBus
    step_index: int

    intent: str
    goals: list[str]

    start_time: float
    state: Literal[
        "new",
        "planned",
        "working",
        "present",
        "done",
        "replan",
        "error",
    ]

    artifacts: list

    def __init__(self, query: str) -> None:
        self.query = query
        self.id = uuid.uuid4().hex
        self.event_bus = EventBus()
        self.step_index = 0
        self.state = "new"
        self.start_time = time.time()


    def _parse_query():
        