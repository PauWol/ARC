from __future__ import annotations

from src.sandbox.runner import run_python
from src.schema import ToolResult
from src.policy import SandboxPolicy

from src.sandbox.bash import run_bash
from src.sandbox.runner import run_python


def run_python_file(file_path: str, policy: SandboxPolicy | None = None) -> ToolResult:
    """
    Read *file_path* and execute its contents in the Python sandbox.

    Parameters
    ----------
    file_path:
        Path to the ``.py`` file to run.
    policy:
        Optional :class:`~policy.SandboxPolicy`.  When *None* the runner
        falls back to :data:`~policy.DEFAULT_POLICY` (or the env-var
        override).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as exc:
        return ToolResult(False, f"error: {str(exc)[:200]}", {"error": str(exc)})

    result = run_python(text, policy=policy)

    summary = "running python file code failed"
    if result.success:
        summary = "ran python file code successfully"

    return ToolResult(result.success, summary, result.data)


__all__ = ["run_python_file", "run_bash", "run_python"]
