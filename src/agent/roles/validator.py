import json
from dataclasses import dataclass
from typing import Optional
from typing import Literal

from sympy.polys.polyconfig import query

from src.agent.llama_runtime import LlamaRuntime
from src.agent.memory import AgentState
from src.agent.roles import build_validator_prompt
from src.agent.types import BaseRole
from src.agent.schema import ToolResult
from src.agent.roles.planner import Plan


VALIDATE_PROMPT = """
ROLE: task validator

TASK:
Decide the current status of the task using the latest result, state, and artifacts.

RULES:
- Use only provided information
- Focus on whether the result completes the task goal
- If tool failed or is unknown → replan
- Prefer marking done only when clearly complete

STATUS:
- done: task is complete
- present: output is ready for user
- continue: more work needed
- replan: change approach needed
"""


@dataclass(slots=True)
class ValidationResult:
    status: Literal["done", "present", "continue", "replan"]
    reason: str
    missing: Optional[list[str]] = None

    def __post_init__(self):
        if self.missing is None:
            self.missing = []


class Validator(BaseRole[ValidationResult]):
    system_prompt = VALIDATE_PROMPT
    output_schema = ValidationResult

    def __init__(
        self, runtime: LlamaRuntime, tokens: int = 100, temperature: float = 0
    ) -> None:
        super().__init__(runtime, tokens, temperature)

    async def run(self, query: str) -> T:
        query = self.build_validator_prompt()

        return await super().run(query)

    def build_validator_prompt(result: ToolResult, plan: Plan) -> str:
        artifact_block = (
            [
                {
                    "type": a.type,
                    "name": a.name,
                    "description": a.description,
                    "tags": a.tags[:4] if a.tags else [],
                    "path": a.path,
                }
                for a in result.artifacts
            ]
            if result.artifacts
            else "No artifacts returned"
        )

        return f"""
    GOAL:
    {state.intent or state.cleaned_input or state.raw_input}

    STATE:
    status: {state.status}
    step_index: {state.step_index}
    error: {state.error or ""}

    CONTEXT:
    {state.compact_prompt()}

    LAST_ACTION:
    tool: {plan.tool}
    reason: {plan.reason}
    input: {json.dumps(plan.input, ensure_ascii=False, separators=(",", ":"))}

    LAST_RESULT:
    success: {str(result.success).lower()}
    summary: {result.summary}
    data: {json.dumps(result.data, ensure_ascii=False, separators=(",", ":"))}

    RESULT_ARTIFACTS:
    {json.dumps(artifact_block, ensure_ascii=False, indent=2)}
    """
