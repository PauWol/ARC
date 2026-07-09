from __future__ import annotations

import asyncio
from typing import Callable

from src.llama_runtime import LlamaRuntime
from src.roles.generator import GeneralGenerator, GeneratedContent, GenerationKind
from src.schema import Artifact, ToolResult
from src.tools.sandbox import (
    run_python_file as _run_python_file,
    run_python as _run_python,
)
from src.tools.sandbox.bash import run_bash as _run_bash
from src.tools.sandbox.policy import SandboxPolicy, policy_from_env


def _active_policy(override: SandboxPolicy | None) -> SandboxPolicy:
    return override if override is not None else policy_from_env()


async def read_file(path: str) -> ToolResult:
    try:
        def _read() -> str:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        text = await asyncio.to_thread(_read)
        return ToolResult(success=True, summary=f"read {path}", data={"text": text})
    except Exception as exc:
        return ToolResult(success=False, summary=f"read_file failed: {exc}")


async def write_file(path: str, content: str) -> ToolResult:
    try:
        def _write() -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        await asyncio.to_thread(_write)

        art = Artifact(
            type="text",
            name=path,
            path=path,
            description="written file",
            created_by="write_file",
        )
        return ToolResult(success=True, summary=f"wrote {path}", artifacts=[art])
    except Exception as exc:
        return ToolResult(success=False, summary=f"write_file failed: {exc}")


def make_default_generator_tool(
    runtime: LlamaRuntime,
    *,
    tokens: int = 4096,
    temperature: float = 0.0,
    default_kind: GenerationKind = "text",
    default_language: str | None = None,
) -> Callable:
    generator = GeneralGenerator(
        runtime=runtime,
        tokens=tokens,
        temperature=temperature,
        default_kind=default_kind,
        default_language=default_language,
    )

    async def generate(
        query: str,
        kind: GenerationKind = "text",
        language: str | None = None,
        filename: str | None = None,
    ) -> ToolResult:
        try:
            result: GeneratedContent = await generator.run(
                query=query,
                kind=kind,
                language=language,
                filename=filename,
            )

            artifacts: list[Artifact] = []
            if result.filename:
                artifacts.append(
                    Artifact(
                        type="text",
                        name=result.filename,
                        path=result.filename,
                        description="generated content",
                        created_by="generator",
                    )
                )

            return ToolResult(
                success=True,
                summary=f"generated {result.kind}",
                data={
                    "kind": result.kind,
                    "filename": result.filename,
                    "content": result.content,
                    "description": result.description,
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            return ToolResult(success=False, summary=f"generator failed: {exc}")

    return generate


def _make_execute_python(policy: SandboxPolicy | None) -> Callable:
    active = _active_policy(policy)

    async def execute_python(
        code: str | None = None,
        file_path: str | None = None,
    ) -> ToolResult:
        provided = sum(x is not None for x in (code, file_path))

        if provided != 1:
            return ToolResult(
                success=False,
                summary="Provide exactly one of: code or file_path",
            )

        if code is not None:
            if not code.strip():
                return ToolResult(success=False, summary="code is empty")
            return await asyncio.to_thread(_run_python, code, active)

        if not file_path.strip():  # type: ignore[union-attr]
            return ToolResult(success=False, summary="file_path is empty")
        return await asyncio.to_thread(_run_python_file, file_path, active)  # pyright: ignore[reportArgumentType]

    return execute_python


def _make_run_bash(policy: SandboxPolicy | None) -> Callable:
    active = _active_policy(policy)

    async def run_bash(
        command: str,
        cwd: str | None = None,
        timeout: float = 10.0,
        confirmed: bool = False,
    ) -> ToolResult:
        return await asyncio.to_thread(
            _run_bash,
            command,
            active,
            cwd,
            timeout,
            confirmed,
        )

    return run_bash


def make_builtin_tools(
    runtime: LlamaRuntime | None = None,
    policy: SandboxPolicy | None = None,
) -> list[Callable]:
    tools: list[Callable] = [
        read_file,
        write_file,
        _make_execute_python(policy),
        _make_run_bash(policy),
    ]

    if runtime is not None:
        tools.append(make_default_generator_tool(runtime))

    return tools


BUILTIN_TOOLS: list[Callable] = make_builtin_tools()