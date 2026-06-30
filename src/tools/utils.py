from typing import Any

from src.schema import ToolResult


class ToolResultError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def tool_result_validator(result: Any, tool_name: str):
    if not isinstance(result, ToolResult):
        raise ToolResultError(
            f"Missing ToolResult return for executed function {tool_name}!\nPlease make it return ToolResult or raise an Error."
        )
