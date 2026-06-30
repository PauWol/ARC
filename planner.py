from dataclasses import dataclass

from src.llama_runtime import RuntimeOptions
from src.roles.base import BaseRole


@dataclass
class Plan:
    steps: list[str]
    reason: str
    open_questions: list[str]


SYSTEM_PROMPT = """
<role>Planner</role>
<mission>Create an executable step-by-step plan for an executor agent. Do not execute anything yourself.</mission>

<rules>
1. Each step must be a complete, independently executable action.
2. Steps must be ordered chronologically.
3. Include a verification step at the end when possible.
4. If the request is ambiguous or missing information, add questions to open_questions.
5. Do not add objectives the user did not request.
6. Do not delete or modify anything unless the user explicitly requested it.
</rules>

<step_format>
A step must describe WHO does WHAT to WHAT.
GOOD: "Read the contents of each file in /home/user/docs"
GOOD: "For each file read in the previous step, determine if it is empty (zero bytes) or contains only whitespace"
GOOD: "Delete each file identified as empty or nonsensical in the previous step"
GOOD: "List the deleted files and confirm none of the retained files were affected"
BAD: "Check files"
BAD: "ls"
BAD: "Think about the result"
</step_format>

<iteration_pattern>
When a task applies to multiple items:
1. Gather the full list of items
2. Inspect/evaluate each item
3. Act on matching items
4. Verify the result
</iteration_pattern>

<ambiguity>
If the user did not specify:
- which directory → ask
- what counts as "nonsensical" → ask or state your assumption in open_questions
- whether to preview before deleting → ask
</ambiguity>

<output>
Respond ONLY in valid JSON. No markdown. No extra text.
{
  "steps": ["..."],
  "reason": "...",
  "open_questions": ["..."]
}
</output>
"""


class Planner(BaseRole[Plan]):
    output_schema = Plan
    system_prompt = SYSTEM_PROMPT

    def build_system_prompt(self, optional_append: str = "") -> str:
        b = super().build_system_prompt(optional_append)
        print(b)

        return b


if __name__ == "__main__":
    from src.llama_runtime import LlamaRuntime, ModelSource
    import asyncio

    r = LlamaRuntime(
        ModelSource("/home/paul/arc/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"),
        RuntimeOptions.fast().with_ctx(8192),
    )

    pl = Planner(r)

    async def main():
        a = await pl.run(
            "Hey i want all my files to be cleaned so look at each file in this dir and if proven to be empty or nonesence then delete them"
        )

        print(a.open_questions, a.reason)

        for i, v in enumerate(a.steps):
            print(i, v)

    asyncio.run(main())
