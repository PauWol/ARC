from dataclasses import dataclass
from src.llama_runtime import LlamaRuntime
from src.roles.base import BaseRole

CODE_PROMPT = """
ROLE: python code generator

TASK:
You generate complete, executable Python code based on user query.
Produce working Python implementations with no explanations.
    
RULES:
- Produce working Python code
- No explanations or commentary
- Code must be complete implementation
- Do not omit imports or dependencies
- Must be executable as-is
- Keep description concise and focus on mainly on code generation
"""
# TODO: Maybe remove must be executable as-is -> dependent on query not always the best


@dataclass(slots=True)
class GeneratedCode:
    filename: str
    code: str
    description: str


class PythonCodeGenerator(BaseRole[GeneratedCode]):
    output_schema = GeneratedCode
    system_prompt = CODE_PROMPT

    def __init__(
        self, runtime: LlamaRuntime, tokens: int = 4096, temperature: float = 0
    ) -> None:
        super().__init__(runtime, tokens, temperature)

    async def run(self, query: str):
        response = await super().run(query)

        if not response.filename.endswith(".py"):
            response.filename += ".py"

        return response
