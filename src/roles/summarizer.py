from dataclasses import dataclass
from src.llama_runtime import LlamaRuntime
from src.roles.base import BaseRole

MEMORY_SUMMARY_PROMPT = """
ROLE: memory compressor

TASK:
Reduce working memory size while preserving important information.

RULES:
- Keep only explicit information from the input
- Remove duplicates and low-value details
- Use short factual statements
- Do not infer goals, plans, next steps, or intentions
- Do not generate explanations
- Do not add information not present in the input
- Preserve important facts, results, constraints, and assets

CATEGORIZATION:
- state: current known facts and status
- key_info: important details and findings
- assets: referenced resources and artifacts
"""
# TODO: Maybe change to artifact-names if output tends to be the whole artifact


@dataclass(slots=True)
class MemorySummary:
    state: list[str]
    key_info: list[str]
    assets: list[str]


class MemorySummarizer(BaseRole[MemorySummary]):
    output_schema = MemorySummary
    system_prompt = MEMORY_SUMMARY_PROMPT

    def __init__(
        self, runtime: LlamaRuntime, tokens: int = 256, temperature: float = 0
    ) -> None:
        super().__init__(runtime, tokens, temperature)

    async def run(self, query: str):
        response = await super().run(query)

        return response

    # TODO: Maybe add guardrails to prevent generating more (larger) or false summaries as the input was
