from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, get_args, get_origin, get_type_hints
import inspect
import re


def _type_to_str(tp: Any) -> str:
    """Convert a Python type annotation into a readable schema type string."""
    if tp is inspect._empty or tp is Any:
        return "any"

    origin = get_origin(tp)
    args = get_args(tp)

    # Optional / Union
    if origin is None:
        name = getattr(tp, "__name__", None)
        return name or str(tp).replace("typing.", "")

    if origin is list:
        inner = _type_to_str(args[0]) if args else "any"
        return f"list[{inner}]"

    if origin is dict:
        key_t = _type_to_str(args[0]) if len(args) > 0 else "any"
        val_t = _type_to_str(args[1]) if len(args) > 1 else "any"
        return f"dict[{key_t}, {val_t}]"

    if origin is tuple:
        inner = ", ".join(_type_to_str(a) for a in args) if args else "any"
        return f"tuple[{inner}]"

    if origin is set:
        inner = _type_to_str(args[0]) if args else "any"
        return f"set[{inner}]"

    if str(origin).endswith("Union"):
        parts = [_type_to_str(a) for a in args]
        return " | ".join(parts)

    return str(tp).replace("typing.", "")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """
    Parse a simple Google-style / NumPy-style docstring.

    Returns:
        (summary, param_descriptions)
    """
    if not doc.strip():
        return "", {}

    lines = [line.rstrip() for line in doc.splitlines()]
    summary = next((line.strip() for line in lines if line.strip()), "")

    param_desc: dict[str, str] = {}
    in_params = False
    current_name: str | None = None

    for raw in lines:
        line = raw.rstrip()

        if re.match(r"^(Args|Arguments|Parameters)\s*:\s*$", line.strip()):
            in_params = True
            current_name = None
            continue

        if in_params and re.match(
            r"^(Returns|Yields|Raises|Examples)\s*:\s*$", line.strip()
        ):
            break

        if not in_params:
            continue

        # Match: name: description
        m = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]+\))?\s*:\s*(.*)$", line)
        if m:
            current_name = m.group(1)
            if not current_name:
                continue
            param_desc[current_name] = m.group(2).strip()
            continue

        # Continuation lines
        if current_name and line.strip():
            param_desc[current_name] += " " + line.strip()

    return summary, param_desc


def generate_tool_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """
    Build a schema dictionary for exactly one function.
    The schema is derived from:
      - function name
      - type hints / signature
      - docstring
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    doc = inspect.getdoc(fn) or ""
    summary, param_desc = _parse_docstring(doc)

    schema: dict[str, Any] = {
        "name": fn.__name__,
        "description": summary or f"Tool function {fn.__name__}",
        "parameters": [],
        "returns": _type_to_str(hints.get("return", sig.return_annotation)),
    }

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        ann = hints.get(name, param.annotation)
        is_required = param.default is inspect._empty
        default_value = None if is_required else param.default

        schema["parameters"].append(
            {
                "name": name,
                "type": _type_to_str(ann),
                "required": is_required,
                "default": default_value,
                "description": param_desc.get(name, ""),
            }
        )

    return schema


def render_tool_prompt(schema: dict[str, Any]) -> str:
    """Render one tool schema as a prompt block."""
    lines: list[str] = []
    lines.append(f"TOOL_NAME: {schema['name']}")
    lines.append(f"DESCRIPTION: {schema['description']}")
    lines.append("PARAMETERS:")

    params = schema.get("parameters", [])
    if not params:
        lines.append("- none")
    else:
        for p in params:
            required = (
                "required" if p["required"] else f"optional (default={p['default']!r})"
            )
            desc = p["description"] or "No description provided."
            lines.append(f"- {p['name']}: {p['type']} | {required} | {desc}")

    # ret = schema.get("returns", "any")
    # lines.append(f"RETURNS: {ret}")
    return "\n".join(lines)


@dataclass(slots=True)
class ToolPromptGenerator:
    functions: list[Callable[..., Any]] = field(default_factory=list)

    def build_prompt(self) -> str:
        """Builds the tool prompt part of the system prompt."""
        if len(self.functions) == 0:
            return ""

        blocks = []
        for fn in self.functions:
            schema = generate_tool_schema(fn)
            blocks.append(render_tool_prompt(schema))

        return "AVAILABLE_TOOLS:\n\n" + "\n\n".join(blocks)

    @classmethod
    def empty(cls):
        """Return an empty class instance for default (not set) usage."""
        return cls([])

    @classmethod
    def from_functions(cls, functions: list[Callable[..., Any]]):
        """Helper to instantiate the Generator directly from a function list."""
        return cls(functions)


# TODO: Add aks question tool etc combined with permission questions
