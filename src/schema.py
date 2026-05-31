from dataclasses import dataclass

INLINE_ARTIFACT_MAX_CHARS = 4000


@dataclass(slots=True)
class Artifact:
    type: str
    name: str

    content: str | None = None
    path: str | None = None

    description: str = ""
