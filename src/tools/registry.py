# registry.py
from __future__ import annotations

import hashlib
import inspect
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sentence_transformers import SentenceTransformer

# ── constants ─────────────────────────────────────────────────────────────────

_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_CACHE = Path(".registry_embeddings.db")


# ── helpers ───────────────────────────────────────────────────────────────────


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity.  Vectors pre-normalized → plain dot product."""
    return float(np.dot(a, b))


# ── embedding store ───────────────────────────────────────────────────────────


class EmbeddingStore:
    """
    SQLite-backed cache mapping SHA-256(text) → float32 embedding.

    The SentenceTransformer model is loaded lazily on the first call to
    ``embed()`` so importing the module has zero warm-up cost.
    """

    def __init__(
        self,
        path: Path = _DEFAULT_CACHE,
        model_name: str = _MODEL_NAME,
    ) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._conn = sqlite3.connect(path,    check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(hash TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        self._conn.commit()

    # ── public ────────────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Return a normalized float32 embedding, hitting the cache first."""
        key = _sha256(text)

        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE hash = ?", (key,)
        ).fetchone()

        if row:
            return np.frombuffer(row[0], dtype=np.float32)

        vec = self._encode(text)
        self._conn.execute(
            "INSERT INTO embeddings (hash, vector) VALUES (?, ?)",
            (key, vec.tobytes()),
        )
        self._conn.commit()
        return vec

    def close(self) -> None:
        self._conn.close()

    # ── private ───────────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode(self, text: str) -> np.ndarray:
        vec = self._get_model().encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)


# ── data types ────────────────────────────────────────────────────────────────


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
    kind: str          # "tool" | "artifact"
    name: str
    score: float
    description: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ref: Any = None    # ToolSpec | Artifact


# ── registry ──────────────────────────────────────────────────────────────────


class ToolRegistry:
    """
    Registers tools and artifacts, and retrieves them by semantic similarity.

    Usage
    -----
    registry = ToolRegistry()

    @registry.tool(description="Add two numbers", tags=("math",))
    def add(a: float, b: float) -> float:
        return a + b

    hits   = registry.search("arithmetic calculation", top_k=3)
    ctx    = registry.build_tool_context("arithmetic calculation")
    result = registry.call("add", a=1, b=2)
    """

    def __init__(
        self,
        cache_path: Path = _DEFAULT_CACHE,
        model_name: str = _MODEL_NAME,
    ) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._artifacts: list[Artifact] = []
        self._store = EmbeddingStore(cache_path, model_name)

    # ── registration ──────────────────────────────────────────────────────────

    def tool(
        self,
        name: str | None = None,
        description: str = "",
        *,
        tags: tuple[str, ...] = (),
        is_generator: bool = False,
    ) -> Callable:
        """Decorator that registers a function as a named tool."""

        def decorator(fn: Callable[..., Any]) -> Callable:
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

    # ── retrieval ─────────────────────────────────────────────────────────────

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def list_artifacts(self) -> list[Artifact]:
        return list(self._artifacts)

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a registered tool by name."""
        return self._tools[name].func(*args, **kwargs)

    # ── semantic search ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        kind: str | None = None,   # "tool" | "artifact" | None (both)
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """
        Return the top-k tools and/or artifacts most similar to *query*.

        Parameters
        ----------
        query:      Natural-language search string.
        top_k:      Maximum number of results.
        kind:       Filter to "tool" or "artifact"; None returns both.
        min_score:  Discard hits below this cosine similarity threshold.
        """
        qv = self._store.embed(query)
        hits: list[SearchHit] = []

        if kind in (None, "tool"):
            for spec in self._tools.values():
                score = _cosine(qv, self._store.embed(spec.search_text()))
                if score >= min_score:
                    hits.append(
                        SearchHit(
                            kind="tool",
                            name=spec.name,
                            score=score,
                            description=spec.description,
                            data={
                                "tags": list(spec.tags),
                                "is_generator": spec.is_generator,
                            },
                            ref=spec,
                        )
                    )

        if kind in (None, "artifact"):
            for art in self._artifacts:
                score = _cosine(qv, self._store.embed(art.search_text()))
                if score >= min_score:
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
                            ref=art,
                        )
                    )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def build_tool_context(self, query: str, *, top_k: int = 5) -> str:
        """
        Render the top-k search results as a compact prompt block consumed by
        the planner (think.py).
        """
        t_out = []
        o_out = []

        hits = self.search(query, top_k=top_k)
        if not hits:
            return "relevant: none"

        lines = ["relevant:"]
        for hit in hits:
            if hit.kind == "tool":
                spec: ToolSpec = hit.ref
                sig = inspect.signature(spec.func)
                params = [
                    f"{n}: {p.annotation if p.annotation is not inspect.Parameter.empty else 'Any'}"
                    for n, p in sig.parameters.items()
                ]
                extra = (
                    f"tags={list(spec.tags)} "
                    f"generator={spec.is_generator} "
                    f"params={params}"
                )

                t_out.append(hit)
            else:
                extra = (
                    f"type={hit.data.get('type', '')} "
                    f"path={hit.data.get('path', '')}"
                )
                o_out.append(hit)
            lines.append(
                f"- {hit.kind}: {hit.name} | {hit.description} | {extra}"
            )
        return "\n".join(lines), t_out,o_out

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the embedding cache."""
        self._store.close()

    def __enter__(self) -> "ToolRegistry":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()