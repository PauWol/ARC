from src.tools.sandbox.runner import run_python
from src.tools.registry import ToolResult


def run_python_file(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    except Exception as e:
        return ToolResult(False, f"error: {str(e)[:200]}", {"error": e})

    r = run_python(text)

    s = "running python file code failed"

    if r.success:
        s = "ran python file code successfully"

    return ToolResult(r.success, s, r.data)


__all__ = ["run_python", "run_python_file"]
