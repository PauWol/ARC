from __future__ import annotations

import json
from typing import Any

from src.llama_runtime import LlamaRuntime
from src.memory import AgentState
from src.assets import json_grammar

THINK_PROMPT = """
You are the step planner of an agent system.

Choose exactly one next action.

You will receive:
- state information
- relevant tools
- relevant artifacts

Rules:
- output ONLY valid JSON
- choose exactly one tool
- prefer the smallest useful step
- use tool_search when you need more relevant tools or artifacts
- use memory_write only to store short important facts
- use finish when the task is complete
- keep reason under 12 words
- never output explanations outside JSON

CRITICAL RULE:
- You NEVER write full programs
- You NEVER output code
- You ONLY describe intent for tools
- Code generation happens ONLY inside tools
- You MUST only use tools listed in "relevant:" section.
- Never invent tool names.
- If no tool fits, respond with: tool: none

Return this JSON shape:
{
  "tool": "tool_name",
  "reason": "short reason",
  "input": {},
  "memory_write": {
    "key": "optional_key",
    "value": "optional_value"
  },
  "done": false
}
"""


def parse_content(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {
            "tool": "tool_search",
            "reason": "fallback to search context",
            "input": {"query": ""},
            "memory_write": {},
            "done": False,
        }

    tool = str(data.get("tool", "tool_search")).strip()
    reason = str(data.get("reason", "")).strip()[:80]
    inp = data.get("input", {})
    mem = data.get("memory_write", {})
    done = bool(data.get("done", False))

    if not isinstance(inp, dict):
        inp = {}

    if not isinstance(mem, dict):
        mem = {}

    return {
        "tool": tool,
        "reason": reason,
        "input": inp,
        "memory_write": {
            "key": str(mem.get("key", "")).strip()[:64],
            "value": mem.get("value", ""),
        },
        "done": done,
    }


def think(runtime: LlamaRuntime, state: AgentState, context: str) -> dict[str, Any]:
    response = runtime.chat(
        messages=[
            {"role": "system", "content": THINK_PROMPT.strip()},
            {
                "role": "user",
                "content": context,
            },
        ],
        grammar=json_grammar(),
        temperature=0.0,
        top_p=0.1,
        max_tokens=500,
        reset=True,
    )

    content = response["choices"][0]["message"]["content"]
    print(content)
    return parse_content(content)
