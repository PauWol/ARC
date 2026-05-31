# validator.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.llama_runtime import LlamaRuntime
from src.memory import AgentState
from src.registry import ToolResult, Artifact
from src.assets import json_grammar


VALIDATE_PROMPT = """
You are the validator of an agent system.

Your job:
- inspect the latest tool result and current state
- decide whether the task is complete
- decide whether the planner should replan
- do NOT choose the next tool directly

Return ONLY valid JSON in this exact shape:
{
  "done": false,
  "status": "continue|replan|present|complete",
  "reason": "short reason",
  "missing": ["short item"],
  "needs_presentation": false
}

Rules:
- done=true only if the task is fully complete
- use "present" when the result is ready for final user output but not yet presented
- use "replan" when the current path failed or is insufficient
- use "continue" when more work is needed
- keep reason under 12 words
- max 3 missing items
- no explanations outside JSON
- If last_success is false, status must be "replan"
- If the last_result contains "unknown tool", status must be "replan"
- Do not call a failed or unknown tool "continue" or "complete"
- done=true only if the task is fully complete
"""


@dataclass(slots=True)
class ValidationResult:
    done: bool = False
    status: str = "continue"
    reason: str = ""
    missing: list[str] = field(default_factory=list)
    needs_presentation: bool = False


def parse_validation(content: str) -> ValidationResult:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ValidationResult(
            done=False,
            status="replan",
            reason="invalid validator output",
        )

    done = bool(data.get("done", False))
    status = str(data.get("status", "continue")).strip().lower()
    reason = str(data.get("reason", "")).strip()[:120]
    needs_presentation = bool(data.get("needs_presentation", False))

    if status not in {"continue", "replan", "present", "complete"}:
        status = "replan"

    missing_raw = data.get("missing", [])
    missing: list[str] = []
    if isinstance(missing_raw, list):
        for item in missing_raw[:3]:
            value = str(item).strip()
            if value:
                missing.append(value[:80])

    return ValidationResult(
        done=done,
        status=status,
        reason=reason,
        missing=missing,
        needs_presentation=needs_presentation,
    )


def _artifact_block(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return "none"

    lines: list[str] = []
    for art in artifacts[-5:]:
        lines.append(
            f"- {art.type}: {art.name} | {art.description} | {art.path or 'inline'}"
        )
    return "\n".join(lines)


def validate(
    runtime: LlamaRuntime,
    state: AgentState,
    action: dict[str, Any],
    result: ToolResult,
    artifacts: list[Artifact],
    fast_path_enabled: bool = True
) -> [ValidationResult,bool]:
    """
    Validate the latest step without choosing the next tool.
    """
    tool_name = str(action.get("tool", "")).strip()

    if fast_path_enabled:
        if not result.success:
            summary = (result.summary or "").lower()

            if "unknown tool" in summary:
                return ValidationResult(
                    done=False,
                    status="replan",
                    reason="unknown tool",
                    missing=[f"use a registered tool instead of {tool_name}"],
                ), True

            return ValidationResult(
                done=False,
                status="replan",
                reason="tool failed",
                missing=["fix tool call", "retry with valid input"],
            ), True

    prompt = (
        f"{state.compact_prompt()}\n\n"
        f"last_action: {action.get('tool', '')}\n"
        f"last_reason: {action.get('reason', '')}\n"
        f"last_result: {result.summary}\n"
        f"last_success: {result.success}\n\n"
        f"artifacts:\n{_artifact_block(artifacts)}"
    )

    response = runtime.chat(
        messages=[
            {"role": "system", "content": VALIDATE_PROMPT.strip()},
            {"role": "user", "content": prompt},
        ],
        grammar=json_grammar(),
        temperature=0.0,
        top_p=0.1,
        max_tokens=100,
        reset=True,
    )

    content = response["choices"][0]["message"]["content"]
    return parse_validation(content), False
