from dataclasses import dataclass, field
from typing import Any, Callable

INLINE_ARTIFACT_MAX_CHARS = 4000


# TODO: Make name and system prompt of config actually do something


@dataclass(slots=True)
class AgentResponse:
    output: str
    success: bool
    iterations: int
    tool_calls: int
    tokens_used: int
    execution_time: float

    def __str__(self) -> str:
        return self.output if self.output is not None else ""

    @property
    def text(self):
        """Return :attr: output"""
        return self.output

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "output": self.output,
            "success": self.success,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "execution_time": round(self.execution_time, 2),
        }


@dataclass(slots=True)
class Artifact:
    type: str
    name: str
    content: str | None = None
    path: str | None = None
    description: str = ""
    created_by: str = ""
    tags: tuple[str, ...] = ()

    def search_text(self) -> str:
        parts = [
            self.type,
            self.name,
            self.description,
            self.created_by,
            " ".join(self.tags),
        ]
        if self.content and len(self.content) <= 2000:
            parts.append(self.content)
        if self.path:
            parts.append(self.path)
        return " ".join(p for p in parts if p)


@dataclass(slots=True)
class ToolResult:
    success: bool
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    tags: tuple[str, ...] = ()
    is_generator: bool = False

    def search_text(self) -> str:
        return f"{self.name} {self.description} {' '.join(self.tags)}"


@dataclass(slots=True)
class SearchHit:
    kind: str  # "tool" | "artifact"
    name: str
    score: float
    description: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ref: Any = None  # ToolSpec | Artifact


@dataclass(slots=True)
class SubAgentResult:
    type: str
    name: str

    content: str | None = None
    path: str | None = None
    summary: str | None = None

    description: str = ""
    artifacts: list[Artifact] | None = None
