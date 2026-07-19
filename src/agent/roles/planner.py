from dataclasses import dataclass, field
from typing import Any, Callable

from src.llama_runtime import LlamaRuntime
from src.types import BaseRole, ReasoningMode, ReasoningSink
from src.tools import ToolPromptGenerator

PLANNER_PROMPT = """
ROLE: planner

TASK:
Select EXACTLY ONE next tool action.
Only one step at a time.
Do not plan multiple steps.

RULES:
- The "tool" field must be EXACTLY one of the names listed under TOOLS below
- copy the name verbatim, do not abbreviate or guess a different name
- Every tool requires an "input" object with its documented arguments;
  never leave "input" empty unless the tool takes no arguments
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
        return {"tool": self.tool, "reason": self.reason, "input": self.input}


class Planner(BaseRole[Plan]):
    output_schema = Plan
    system_prompt: str = PLANNER_PROMPT

    def __init__(
        self,
        runtime: LlamaRuntime,
        tools: list[Callable],
        tokens: int = 500,
        temperature: float = 0,
        system_prompt_addition: str = "",
        # Planning quality benefits from an explicit reasoning pass even on
        # non-thinking models — default "on" rather than "auto" here.
        # Pass reasoning="auto" instead if you'd rather it track whatever
        # the loaded model natively supports.
        reasoning: ReasoningMode = "on",
        reasoning_hook: ReasoningSink | None = None,
    ) -> None:
        self._tool_names = {
            getattr(t, "__name__", t.__class__.__name__).lower() for t in tools
        }
        # Tools whose signature takes no required positional/keyword args
        # (beyond self) don't need a non-empty "input" — best-effort check,
        # extend this set manually for tools with an unusual calling
        # convention that inspect can't see through (e.g. **kwargs-only).
        self._no_arg_tools = {
            getattr(t, "__name__", t.__class__.__name__).lower()
            for t in tools
            if _takes_no_required_args(t)
        }
        self.system_prompt_addition = system_prompt_addition
        super().__init__(
            runtime,
            tokens,
            temperature,
            ToolPromptGenerator.from_functions(tools),
            reasoning=reasoning,
            reasoning_tokens=300,
            reasoning_hook=reasoning_hook,
            role_name="planner",
        )

    def build_system_prompt(self, optional_append: str = "") -> str:
        return super().build_system_prompt(
            optional_append + self.system_prompt_addition
        )

    def _validate(self, plan: Plan) -> str | None:
        tool = str(plan.tool).strip().lower()

        if not tool:
            return 'the "tool" field is empty — pick one of the listed tool names, or "finish"'

        if tool == "finish":
            return None

        if tool not in self._tool_names:
            return (
                f'"{tool}" is not a registered tool name. Valid names: '
                f'{sorted(self._tool_names)}. Copy one verbatim into "tool".'
            )

        if not plan.input and tool not in self._no_arg_tools:
            return (
                f'tool "{tool}" was selected but "input" is empty. '
                "Provide its required arguments as a JSON object."
            )

        return None


def _takes_no_required_args(tool: Callable) -> bool:
    import inspect

    try:
        sig = inspect.signature(tool)
    except TypeError, ValueError:
        return False  # unknown signature — assume it needs input, safer default

    for p in sig.parameters.values():
        if p.default is inspect.Parameter.empty and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return False
    return True
