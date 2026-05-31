# registry.py
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Optional
from collections import Counter
import math
import re

TOKEN_RE = re.compile(r"[a-z0-9_]+")


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
    ref: Any = None  # ToolSpec or Artifact


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._artifacts: list[Artifact] = []

    def tool(
        self,
        name: str | None = None,
        description: str = "",
        *,
        tags: tuple[str, ...] = (),
        is_generator: bool = False,
    ):
        def decorator(fn: Callable[..., Any]):
            tool_name = name or fn.__name__
            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=description or (fn.__doc__ or "").strip(),
                func=fn,
                tags=tags,
                is_generator=is_generator,
            )
            return fn

        return decorator

    def register_artifact(self, artifact: Artifact) -> None:
        self._artifacts.append(artifact)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def list_artifacts(self) -> list[Artifact]:
        return list(self._artifacts)

    def call(self, name: str, *args, **kwargs) -> Any:
        tool = self._tools[name]
        return tool.func(*args, **kwargs)

    def _vectorize(self, text: str) -> Counter[str]:
        tokens = TOKEN_RE.findall(text.lower())
        return Counter(tokens)

    def _cosine(self, a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        for k, av in a.items():
            bv = b.get(k, 0)
            dot += av * bv
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        kind: str | None = None,  # "tool" | "artifact" | None
    ) -> list[SearchHit]:
        qv = self._vectorize(query)
        hits: list[SearchHit] = []

        if kind in (None, "tool"):
            for tool in self._tools.values():
                tv = self._vectorize(tool.search_text())
                score = self._cosine(qv, tv)
                if score > 0:
                    hits.append(
                        SearchHit(
                            kind="tool",
                            name=tool.name,
                            score=score,
                            description=tool.description,
                            data={
                                "tags": list(tool.tags),
                                "is_generator": tool.is_generator,
                            },
                            ref=tool,
                        )
                    )

        if kind in (None, "artifact"):
            for art in self._artifacts:
                av = self._vectorize(art.search_text())
                score = self._cosine(qv, av)
                if score > 0:
                    hits.append(
                        SearchHit(
                            kind="artifact",
                            name=art.name,
                            score=score,
                            description=art.description,
                            data={
                                "type": art.type,
                                "path": art.path,
                                "created_by": art.created_by,
                                "tags": list(art.tags),
                            },
                        )
                    )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def build_tool_context(self, query: str, *, top_k: int = 5) -> str:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return "relevant: none"

        lines = ["relevant:"]
        for hit in hits:
            extra = ""
            if hit.kind == "tool":
                tool: ToolSpec = hit.ref

                sig = inspect.signature(tool.func)

                params = []
                for name, p in sig.parameters.items():
                    anno = p.annotation
                    if anno is inspect._empty:
                        anno = "Any"
                    params.append(f"{name}: {anno}")

                extra = (
                    f"tags={list(tool.tags)} "
                    f"generator={tool.is_generator} "
                    f"params={params}"
                )
            elif hit.kind == "artifact":
                extra = (
                    f"type={hit.data.get('type', '')} path={hit.data.get('path', '')}"
                )
            lines.append(f"- {hit.kind}: {hit.name} | {hit.description} | {extra}")
        return "\n".join(lines)
