from dataclasses import dataclass, field
from src.llama_runtime import LlamaRuntime
from src.types import BaseRole

EXTRACT_PROMPT = """
ROLE: intent extractor

TASK: 
Extract the user's task intent and goals.
Identify what the user wants to achieve and break it into structured intent.

RULES:
- intent max 12 words
- goals max 5 items
- each goal max 6 words
- be concise
- no explanations
- no extra text outside JSON
- Never invent intent or goals
- Facts only
"""

INPUT_TOKEN_THRESHOLD = 150  # 300


@dataclass(slots=True)
class ExtractedTask:
    intent: str = ""
    goals: list[str] = field(default_factory=list[str])


class TaskExtractor(BaseRole[ExtractedTask]):
    output_schema = ExtractedTask
    system_prompt: str = EXTRACT_PROMPT

    def __init__(
        self, runtime: LlamaRuntime, tokens: int = 150, temperature: float = 0
    ) -> None:
        super().__init__(runtime, tokens, temperature)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    async def run(self, query: str):
        if self.estimate_tokens(query) < INPUT_TOKEN_THRESHOLD:
            return ExtractedTask(intent=query)

        return await super().run(query)
