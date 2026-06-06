from src.tools.registry import ToolResult, ToolRegistry, Artifact
from src.tools.sandbox import run_python_file as r_p_f, run_python


def default_tools(registry: ToolRegistry):
    """Register the default Tools."""

    @registry.tool(
        name="tool_search",
        description="Search relevant tools and artifacts by query",
        tags=("search", "lookup"),
        is_generator=False,
    )
    def tool_search(query: str, kind: str | None = None, top_k: int = 5) -> ToolResult:
        try:
            top_k = max(1, min(int(top_k), 20))
            hits = registry.search(query, kind=kind, top_k=top_k)
        except Exception as e:
            return ToolResult(success=False, summary=f"tool search failed: {e}")

        return ToolResult(
            success=True,
            summary=f"found {len(hits)} matches",
            data={
                "hits": [
                    {
                        "kind": h.kind,
                        "name": h.name,
                        "score": h.score,
                        "description": h.description,
                        "data": h.data,
                    }
                    for h in hits
                ]
            },
        )

    @registry.tool(
        name="read_file",
        description="Read text from a file",
        tags=("files", "io"),
        is_generator=False,
    )
    def read_file(path: str) -> ToolResult:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return ToolResult(
                success=True,
                summary=f"read {path}",
                data={"text": text},
            )
        except Exception as e:
            return ToolResult(success=False, summary=f"read_file failed: {e}")

    @registry.tool(
        name="write_file",
        description="Write text to a file",
        tags=("files", "io"),
        is_generator=False,
    )
    def write_file(path: str, content: str) -> ToolResult:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            art = Artifact(
                type="text",
                name=path,
                path=path,
                description="written file",
                created_by="write_file",
            )
            registry.register_artifact(art)

            return ToolResult(
                success=True,
                summary=f"wrote {path}",
                artifacts=[art],
            )
        except Exception as e:
            return ToolResult(success=False, summary=f"write_file failed: {e}")

    def _resolve_artifact_by_name(registry: ToolRegistry, name: str) -> Artifact | None:
        for art in registry.list_artifacts():
            if art.name == name:
                return art
        return None

    def _run_artifact_code(artifact_name: str) -> ToolResult:
        art = _resolve_artifact_by_name(registry, artifact_name)

        if not art:
            return ToolResult(
                success=False,
                summary=f"artifact not found: {artifact_name}",
            )

        if art.path:
            return ToolResult(
                success=False,
                summary=f"artifact is a file, use run_python_file instead: {artifact_name}",
            )

        if not art.content:
            return ToolResult(
                success=False,
                summary=f"artifact has no inline code: {artifact_name}",
            )

        return run_python(art.content)

    @registry.tool(
        name="run_python",
        description=(
            "Execute Python code in a sandbox.\n\n"
            "Examples:\n"
            "- code='print(\"hello\")'\n"
            "- file_path='main.py'\n"
            "- artifact_name='fibonacci_simulation'\n\n"
            "Provide exactly one input source."
        ),
        tags=("python", "execute", "sandbox"),
        is_generator=False,
    )
    def execute_python(
        code: str | None = None,
        file_path: str | None = None,
        artifact_name: str | None = None,
    ) -> ToolResult:
        provided = sum(x is not None for x in (code, file_path, artifact_name))

        if provided != 1:
            return ToolResult(
                success=False,
                summary="Provide exactly one of: code, file_path, artifact_name",
            )

        if code is not None:
            if not code.strip():
                return ToolResult(success=False, summary="code is empty")
            return run_python(code)

        if file_path is not None:
            if not file_path.strip():
                return ToolResult(success=False, summary="file_path is empty")
            return r_p_f(file_path)

        if artifact_name is not None:
            if not artifact_name.strip():
                return ToolResult(success=False, summary="artifact_name is empty")
            return _run_artifact_code(artifact_name)

        return ToolResult(success=False, summary="no execution source provided")

    @registry.tool(
        name="calculator",
        description="Deterministic arithmetic calculator for basic math.",
        tags=("math", "deterministic", "calculator"),
        is_generator=False,
    )
    def calculator(operation: str, operands: list[float]) -> ToolResult:
        try:
            if not operands:
                return ToolResult(success=False, summary="no operands provided")

            if operation in {"+", "add"}:
                return ToolResult(success=True, summary=str(sum(operands)))

            if operation in {"-", "sub"}:
                result = operands[0]
                for value in operands[1:]:
                    result -= value
                return ToolResult(success=True, summary=str(result))

            if operation in {"*", "mul"}:
                result = 1.0
                for value in operands:
                    result *= value
                return ToolResult(success=True, summary=str(result))

            if operation in {"/", "div"}:
                result = operands[0]
                for value in operands[1:]:
                    if value == 0:
                        return ToolResult(success=False, summary="division by zero")
                    result /= value
                return ToolResult(success=True, summary=str(result))

            return ToolResult(
                success=False, summary=f"unsupported operation: {operation}"
            )
        except Exception as e:
            return ToolResult(success=False, summary=f"calculator failed: {e}")
