from dataclasses import dataclass

INLINE_ARTIFACT_MAX_CHARS = 4000


@dataclass(slots=True)
class Artifact:
    type: str
    name: str

    content: str | None = None
    path: str | None = None

    description: str = ""

@dataclass(slots=True)
class SubAgentResult:
    type: str
    name: str

    content: str | None = None
    path: str | None = None
    summary: str| None = None

    description: str = ""
    artifacts: list[Artifact]|None = None