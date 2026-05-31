from src.registry import ToolResult, ToolRegistry, Artifact
from src.sandbox import run_python_file as r_p_f, run_python

def default_tools(registry: ToolRegistry):
    """Register the default Tools."""

    @registry.tool(
        name="tool_search",
        description="Search relevant tools and artifacts by query",
        tags=("search", "lookup"),
    )
    def tool_search(query: str, kind: str | None = None, top_k: int = 5) -> ToolResult:
        hits = registry.search(query, kind=kind, top_k=top_k)
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
        name="read_file", description="Read text from a file", tags=("files", "io")
    )
    def read_file(path: str) -> ToolResult:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return ToolResult(
            success=True,
            summary=f"read {path}",
            data={"text": text},
        )

    @registry.tool(
        name="write_file", description="Write text to a file", tags=("files", "io")
    )
    def write_file(path: str, content: str) -> ToolResult:
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

    @registry.tool(
        name="run_python_code",
        description="Run any python code directly sandboxed",
        tags=("code", "python", "executor"),
        is_generator=False,
    )
    def run_python_code(code: str) -> ToolResult:
        return run_python(code)

    @registry.tool(
        name="run_python_file",
        description="Execute a Python file inside the sandbox by file path. "
        "Use this when the code is stored on disk or written to an artifact file.",
        tags=("python", "execute", "sandbox", "file"),
        is_generator=False,
    )
    def run_python_file(file_path: str) -> ToolResult:
        return r_p_f(file_path)

    def resolve_artifact_by_name(registry: ToolRegistry, name: str) -> Artifact | None:
        for art in registry.list_artifacts():
            if art.name == name:
                return art
        return None

    @registry.tool(
        name="run_artifact_python",
        description="Execute a Python artifact from memory. "
        "Use this only when the artifact contains inline code and does not point to a file.",
        tags=("python", "execute", "sandbox", "artifact"),
        is_generator=False,
    )
    def run_artifact_code(artifact_name: str):
        art = resolve_artifact_by_name(registry, artifact_name)

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
        name="calculator",
        description="Deterministic arithmetic calculator for basic math.",
        tags=("math", "deterministic", "calculator"),
        is_generator=False,
    )
    def calculator(operation: str, operands: list[float]) -> ToolResult:
        if operation in {"+", "add"}:
            return ToolResult(success=True, summary=str(sum(operands)))
        return ToolResult(success=False, summary=f"unsupported operation: {operation}")
