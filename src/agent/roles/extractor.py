from dataclasses import dataclass, field
from src.agent.llama_runtime import LlamaRuntime
from src.agent.types import BaseRole

from src.constants import EXTRACTOR_INPUT_TOKEN_THRESHOLD

EXTRACT_PROMPT = """
ROLE: Memory extractor

TASK: 
Extract the current agent state from the conversation.
Produce a compact working memory that can be updated over time.

RULES:
- Be concise.
- Preserve important technical details.
- Never invent information.
- Ignore greetings, filler, repetition, and conversational noise.
- Exclude completed or obsolete tasks unless they remain relevant.
- Output valid JSON only.
- No explanations or markdown.
"""


@dataclass(slots=True)
class ExtractedMemory:
    intent: str = ""
    goals: list[str] = field(default_factory=list[str])
    constrains: list[str] = field(default_factory=list[str])
    previous_decisions: list[str] = field(default_factory=list[str])
    facts: list[str] = field(default_factory=list[str])


class Extractor(BaseRole[ExtractedMemory]):
    output_schema = ExtractedMemory
    system_prompt: str = EXTRACT_PROMPT

    def __init__(
        self, runtime: LlamaRuntime, tokens: int = 150, temperature: float = 0
    ) -> None:
        super().__init__(runtime, tokens, temperature)

    def token_count(self, text: str) -> int:
        return self.runtime.count_tokens(text)

    async def run(self, query: str):
        if self.token_count(query) < int(EXTRACTOR_INPUT_TOKEN_THRESHOLD):
            return ExtractedMemory(intent=query)

        return await super().run(query)
