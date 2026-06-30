from dataclasses import dataclass, field
from typing import Any, Callable

from src.llama_runtime import LlamaRuntime
from src.roles.base import BaseRole
from src.tools import ToolPromptGenerator

ACTION_PROMPT = """
ROLE: planner

TASK:
Select EXACTLY ONE next tool action.
Only one step at a time.
Do not plan multiple steps.
Prefer bash if possible

RULES:
- Output ONLY valid JSON
- One tool call per step
- Never invent tools
- If task is complete → use "finish"
- If no tool is needed → use "finish"
"""


@dataclass(slots=True)
class Plan:
    tool: str = ""
    reason: str = ""
    input: dict[str, Any] = field(default_factory=dict)

    @property
    def to_dict(self):
        return {
            "tool": self.tool,
            "reason": self.reason,
            "input": self.input,
        }


class Planner(BaseRole[Plan]):
    output_schema = Plan
    system_prompt: str = ACTION_PROMPT

    def __init__(
        self,
        runtime: LlamaRuntime,
        tools: list[Callable],
        tokens: int = 500,
        temperature: float = 0,
    ) -> None:
        super().__init__(
            runtime, tokens, temperature, ToolPromptGenerator.from_functions(tools)
        )
