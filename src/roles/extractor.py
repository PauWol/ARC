from __future__ import annotations

import json
from dataclasses import dataclass, field
from src.llama_runtime import LlamaRuntime
from src.assets import json_grammar


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

INPUT_TOKEN_THRESHOLD = 300


@dataclass(slots=True)
class ExtractedTask:
    intent: str = ""
    goals: list[str] = field(default_factory=list[str])


class TaskExtractor:
    """Tiny structured extractor for intent and goals."""

    def __init__(self, runtime: LlamaRuntime):
        self.runtime: LlamaRuntime = runtime

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    def extract(self, text: str) -> ExtractedTask:
        """Extract intent and goals from user input."""

        if self.estimate_tokens(text) < INPUT_TOKEN_THRESHOLD:
            return ExtractedTask(intent=text)

        response = self.runtime.chat(
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT.strip()},
                {"role": "user", "content": text},
            ],
            grammar=json_grammar(),
            temperature=0.0,
            top_p=0.1,
            max_tokens=120,
            reset=True,
        )

        content: str = response["choices"][0]["message"]["content"]

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return ExtractedTask()

        intent: str = str(data.get("intent", "")).strip()
        goals_raw: str = data.get("goals", [])

        goals: list[str] = []
        if isinstance(goals_raw, list):
            for item in goals_raw[:5]:
                value = str(item).strip()
                if value:
                    goals.append(value[:60])

        return ExtractedTask(intent=intent[:120], goals=goals)
