from __future__ import annotations

from enum import StrEnum
import re
from dataclasses import dataclass, field
import time
from typing import Any, Literal
import uuid

from src.agent.events import EventBus
from src.agent.roles import ExtractedMemory, Extractor
from src.agent.roles.compressor import CompressedMemory, Compressor
from src.agent.schema import Artifact


_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_LINE_RE = re.compile(r"^(.*?)(?:\n\1)+$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _dedupe_lines(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []

    for line in text.splitlines():
        line = normalize_text(line)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)

    return "\n".join(out)


def _dedupe_sentences(text: str) -> str:
    parts = _SENTENCE_SPLIT_RE.split(text)
    seen: set[str] = set()
    out: list[str] = []

    for part in parts:
        part = normalize_text(part)
        if not part:
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)

    return " ".join(out)


def extract_user_query(text: str) -> str:
    """
    Return a cleaned user query:
    - removes repeated lines/sentences
    - normalizes whitespace
    """
    text = _dedupe_lines(text)
    text = _dedupe_sentences(text)
    return normalize_text(text)


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


class SessionState(StrEnum):
    NEW = "new"
    PLANNED = "planned"
    WORKING = "working"
    PRESENT = "present"
    DONE = "done"
    REPLAN = "replan"
    ERROR = "error"


class StateSetter:
    def __init__(self, session: "Session") -> None:
        self._session = session

    @property
    def new(self) -> None:
        self._session.state = SessionState.NEW

    @property
    def planned(self) -> None:
        self._session.state = SessionState.PLANNED

    @property
    def working(self) -> None:
        self._session.state = SessionState.WORKING

    @property
    def present(self) -> None:
        self._session.state = SessionState.PRESENT

    @property
    def done(self) -> None:
        self._session.state = SessionState.DONE

    @property
    def replan(self) -> None:
        self._session.state = SessionState.REPLAN

    @property
    def error(self) -> None:
        self._session.state = SessionState.ERROR


@dataclass
class Session:
    query: str
    id: str
    event_bus: EventBus
    step_index: int

    initial_memory: ExtractedMemory

    start_time: float
    state: SessionState
    artifacts: list

    extractor: Extractor

    @classmethod
    async def new(cls, query: str, extractor: Extractor) -> "Session":
        query = extract_user_query(query)
        memory = await extractor.run(query)

        return cls(
            query=query,
            id=uuid.uuid4().hex,
            event_bus=EventBus(),
            step_index=0,
            initial_memory=memory,
            start_time=time.time(),
            state=SessionState.NEW,
            artifacts=[],
            extractor=extractor,
        )

    @property
    def is_done(self):
        return self.state == "done"

    @property
    def set_state(self) -> StateSetter:
        return StateSetter(self)


@dataclass
class WorkingMemory:
    compressor: Compressor
    _mem: dict[str, list[str]] = field(
        default_factory=lambda: {
            "facts": [],
            "results": [],
            "errors": [],
            "temp": [],
        }
    )

    _token_count: int = 0

    def __init__(self, compressor: Compressor) -> None:
        self.compressor = compressor

    def _accumulate_tokens(self, text: object):
        self._token_count += self.compressor.token(text)

    def compress(self):
        pass

    @property
    def token(self):
        return self._token_count
