from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.llama_runtime import LlamaRuntime
from src.types import BaseRole

GenerationKind = Literal[
    "code",
    "text",
    "summary",
    "markdown",
    "json",
    "yaml",
    "config",
]


def build_generation_prompt(kind: str, language: str | None = None) -> str:
    language_line = f"Language: {language}\n" if language else ""

    return f"""
    ```

    ROLE: general content generator

    TASK:
    You generate complete, high-quality output based on the user query.

    OUTPUT KIND:
    {kind}
    {language_line}
    RULES:

    * Produce only the requested output
    * No explanations unless explicitly requested
    * Be complete and useful
    * Follow the requested format strictly
    * Keep the response concise when appropriate
    * If code is requested, include all necessary imports and dependencies
    * If text is requested, write polished final text
    * If a summary is requested, focus on the essential points only
    """.strip()


@dataclass(slots=True)
class GeneratedContent:
    kind: GenerationKind
    filename: str | None = None
    content: str = ""
    description: str = ""


class Generator(BaseRole[GeneratedContent]):
    output_schema = GeneratedContent

    def __init__(
        self,
        runtime: LlamaRuntime,
        tokens: int = 4096,
        temperature: float = 0,
        default_kind: GenerationKind = "text",
        default_language: str | None = None,
    ) -> None:
        self.default_kind = default_kind
        self.default_language = default_language
        self.system_prompt = build_generation_prompt(default_kind, default_language)
        super().__init__(runtime, tokens, temperature)

    async def run(
        self,
        query: str,
        kind: GenerationKind = "text",
        language: str | None = None,
        filename: str | None = None,
    ) -> GeneratedContent:
        active_kind: GenerationKind = kind or self.default_kind
        active_language = language or self.default_language

        old_system_prompt = self.system_prompt
        self.system_prompt = build_generation_prompt(active_kind, active_language)

        try:
            response = await super().run(query)
        finally:
            self.system_prompt = old_system_prompt

        if active_kind == "code" and filename:
            if "." not in filename and active_language:
                filename = f"{filename}.{active_language}"
            response.filename = filename
        elif active_kind == "code" and response.filename:
            if "." not in response.filename:
                ext = active_language or "py"
                response.filename = f"{response.filename}.{ext}"

        response.kind = active_kind
        return response
