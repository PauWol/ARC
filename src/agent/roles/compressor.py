from dataclasses import dataclass, field
from src.agent.llama_runtime import LlamaRuntime
from src.agent.types import BaseRole

COMPRESS_PROMPT = """
Compress this memory.

Remove duplicates, repetition, filler, obsolete info, and completed tasks.
Keep active tasks, goals, constraints, decisions, facts, preferences, pending work, and technical details.
Preserve meaning. Never invent.
Output only the compressed memory.
"""


@dataclass(slots=True)
class CompressedMemory:
    working_memory: dict[str, list[str]] = field(
        default_factory=lambda: {
            "facts": [],
            "results": [],
            "errors": [],
            "temp": [],
        }
    )


class Compressor(BaseRole[CompressedMemory]):
    output_schema = CompressedMemory
    system_prompt: str = COMPRESS_PROMPT

    def __init__(
        self, runtime: LlamaRuntime, tokens: int = 650, temperature: float = 0
    ) -> None:
        super().__init__(runtime, tokens, temperature)

    def token(self, _inp):
        return self.runtime.count_tokens(_inp)
