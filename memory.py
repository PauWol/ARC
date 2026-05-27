from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── cleanup patterns ──────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]{3,}`")
_URL_RE = re.compile(r"https?://[^\s\'\"<>()\[\]]+|www\.[^\s\'\"<>()\[\]]+", re.I)
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\[\w\\.\-/]+)"  # Windows: C:\foo\bar
    r"|(?:/[\w.\-/]+(?:\.\w+)?)"  # Unix absolute: /usr/local/bin
    r"|(?:\.{1,2}/[\w.\-/]+(?:\.\w+)?)",  # Relative: ./foo, ../bar.py
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and trim the string."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_context_noise(text: str) -> str:
    """
    Remove code blocks, inline code, URLs, and file paths from user input.

    This is the cleaned text you can use for intent extraction or prompt packing.
    """
    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _PATH_RE.sub(" ", text)
    return normalize_text(text)


def extract_context_items(text: str) -> list[str]:
    """
    Extract non-core context items such as code, URLs, and paths.

    These are stored separately so the agent can still use them if needed.
    """
    items: list[str] = []
    seen: set[str] = set()

    def add(tag: str, value: str) -> None:
        value = normalize_text(value)
        key = f"{tag}:{value.lower()}"
        if value and key not in seen:
            seen.add(key)
            items.append(f"[{tag}] {value}")

    for match in _CODE_BLOCK_RE.finditer(text):
        add("code", match.group().strip("`"))

    for match in _INLINE_CODE_RE.finditer(text):
        add("code", match.group().strip("`"))

    for match in _URL_RE.finditer(text):
        add("url", match.group())

    for match in _PATH_RE.finditer(text):
        add("path", match.group())

    return items


# ── state ────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AgentState:
    """
    Minimal per-job state for a simple LLM agent.

    This is the short-term / working memory for the current task.
    """

    raw_input: str
    cleaned_input: str = ""

    intent: str = ""
    goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    context_items: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    working_memory: dict[str, Any] = field(default_factory=dict)

    task_type: str = ""
    step_index: int = 0
    status: str = "new"
    done: bool = False
    error: str | None = None

    @classmethod
    def from_user_input(cls, text: str) -> "AgentState":
        """Create initial state from raw user input."""
        return cls(
            raw_input=text,
            cleaned_input=strip_context_noise(text),
            context_items=extract_context_items(text),
        )

    def remember(self, key: str, value: Any) -> None:
        """Store a working-memory value for the current job."""
        self.working_memory[key] = value

    def forget(self, key: str) -> None:
        """Remove one working-memory item if present."""
        self.working_memory.pop(key, None)

    def advance(self, note: str | None = None) -> None:
        """Move to the next step and optionally store a note."""
        self.step_index += 1
        if note:
            self.remember(f"step_{self.step_index}", note)

    def compact_prompt(self) -> str:
        """
        Render the most important state into a short prompt block.

        Useful for tool calls, planner prompts, or executor prompts.
        """
        parts: list[str] = []

        if self.intent:
            parts.append(f"intent: {self.intent}")

        if self.goals:
            parts.append("goals: " + "; ".join(self.goals[:3]))

        if self.constraints:
            parts.append("constraints: " + "; ".join(self.constraints[:3]))

        if self.context_items:
            parts.append("context: " + "; ".join(self.context_items[:4]))

        if self.open_questions:
            parts.append("open: " + "; ".join(self.open_questions[:3]))

        if self.working_memory:
            parts.append(
                "memory: "
                + "; ".join(
                    f"{k}={v}" for k, v in list(self.working_memory.items())[:4]
                )
            )

        return "\n".join(parts)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the state."""
        return {
            "raw_input": self.raw_input,
            "cleaned_input": self.cleaned_input,
            "intent": self.intent,
            "goals": list(self.goals),
            "constraints": list(self.constraints),
            "context_items": list(self.context_items),
            "open_questions": list(self.open_questions),
            "working_memory": dict(self.working_memory),
            "task_type": self.task_type,
            "step_index": self.step_index,
            "status": self.status,
            "done": self.done,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"AgentState(intent={self.intent!r}, goals={self.goals}, "
            f"constraints={self.constraints}, context_items={self.context_items}, "
            f"open_questions={self.open_questions}, step_index={self.step_index}, "
            f"status={self.status!r}, done={self.done})"
        )


# ── tiny agent-side helpers ───────────────────────────────────────────────────


def build_initial_state(query: str) -> AgentState:
    """Convenience helper for creating the first state object."""
    return AgentState.from_user_input(query)


def attach_plan(state: AgentState, plan: list[str]) -> None:
    """Store the current plan in working memory and sync it to state."""
    state.remember("plan", plan)
    state.status = "planned"


def attach_observation(state: AgentState, observation: str) -> None:
    """Store one observation from a tool or model step."""
    state.remember(f"obs_{state.step_index}", observation)
