# extractor.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llama_cpp import LlamaGrammar
from llama_runtime import LlamaRuntime

GRAMMAR_PATH = Path(__file__).with_name("json.gbnf")
JSON_GRAMMAR = LlamaGrammar.from_string(GRAMMAR_PATH.read_text(encoding="utf-8"))

EXTRACT_PROMPT = """
Extract the user's task intent and goals.

Return ONLY valid JSON in this exact shape:
{
  "intent": "short imperative sentence",
  "goals": ["short goal", "short goal"]
}

Rules:
- intent max 12 words
- goals max 5 items
- goals max 6 words each
- concise
- no explanations
"""


@dataclass(slots=True)
class ExtractedTask:
    intent: str = ""
    goals: list[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.goals is None:
            self.goals = []


class TaskExtractor:
    """Tiny structured extractor for intent and goals."""

    def __init__(self, runtime: LlamaRuntime):
        self.runtime = runtime

    def extract(self, text: str) -> ExtractedTask:
        """Extract intent and goals from user input."""
        response = self.runtime.chat(
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT.strip()},
                {"role": "user", "content": text},
            ],
            grammar=JSON_GRAMMAR,
            temperature=0.0,
            top_p=0.1,
            max_tokens=120,
            reset=True,
        )

        content = response["choices"][0]["message"]["content"]

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return ExtractedTask()

        intent = str(data.get("intent", "")).strip()
        goals_raw = data.get("goals", [])

        goals: list[str] = []
        if isinstance(goals_raw, list):
            for item in goals_raw[:5]:
                value = str(item).strip()
                if value:
                    goals.append(value[:60])

        return ExtractedTask(intent=intent[:120], goals=goals)
