"""
Built-in agent tools.

Policy is a **system-level** concern and is never exposed as an LLM-visible
parameter.  Set it once at startup via environment variables:

    SANDBOX_ALLOW=READ,WRITE,EXECUTE,NETWORK
    SANDBOX_CONFIRM=DELETE,SYSTEM,INSTALL

or pass a :class:`~policy.SandboxPolicy` object directly to
:func:`make_builtin_tools` when constructing the agent.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from src.schema import ToolResult, Artifact
from src.tools.sandbox import (
    run_python_file as _run_python_file,
    run_python as _run_python,
)
from src.tools.sandbox.bash import run_bash as _run_bash
from src.tools.sandbox.policy import SandboxPolicy, policy_from_env


# ── helpers ───────────────────────────────────────────────────────────────────


def _active_policy(override: SandboxPolicy | None) -> SandboxPolicy:
    """Return *override* if given, otherwise read from env vars."""
    return override if override is not None else policy_from_env()


# ── file I/O ──────────────────────────────────────────────────────────────────


async def read_file(path: str) -> ToolResult:
    """Read text from a file.

    Args:
        path: Path to the file to read.
    """
    try:

        def _read() -> str:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        text = await asyncio.to_thread(_read)
        return ToolResult(success=True, summary=f"read {path}", data={"text": text})
    except Exception as exc:
        return ToolResult(success=False, summary=f"read_file failed: {exc}")


async def write_file(path: str, content: str) -> ToolResult:
    """Write text to a file.

    Args:
        path: Destination file path.
        content: Text content to write.
    """
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


# ── Python execution ──────────────────────────────────────────────────────────


def _make_execute_python(policy: SandboxPolicy | None) -> Callable:
    """
    Return an ``execute_python`` async function bound to *policy*.

    The returned function is what gets registered as an agent tool — the
    policy is captured in the closure and never appears in the tool schema.
    """
    active = _active_policy(policy)

    async def execute_python(
        code: str | None = None,
        file_path: str | None = None,
    ) -> ToolResult:
        """Execute Python code in a sandbox. Provide exactly one input source.

        Args:
            code: Python source code to run directly.
            file_path: Path to a .py file to execute.
        """
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

        # file_path branch
        if not file_path.strip():  # type: ignore[union-attr]
            return ToolResult(success=False, summary="file_path is empty")
        return await asyncio.to_thread(_run_python_file, file_path, active)  # pyright: ignore[reportArgumentType]

    return execute_python


# ── Bash execution ────────────────────────────────────────────────────────────


def _make_run_bash(policy: SandboxPolicy | None) -> Callable:
    """
    Return a ``run_bash`` async function bound to *policy*.

    ``confirmed`` IS exposed to the LLM — it is the mechanism by which the
    agent re-invokes a command after surfacing a confirmation prompt:

    1. Agent calls ``run_bash(command="rm -rf build/")``
    2. Returns ``ToolResult(success=False, data={"confirmation_required": True, ...})``
    3. Agent surfaces the prompt; user approves
    4. Agent calls ``run_bash(command="rm -rf build/", confirmed=True)``
    """
    active = _active_policy(policy)

    async def run_bash(
        command: str,
        cwd: str | None = None,
        timeout: float = 10.0,
        confirmed: bool = False,
    ) -> ToolResult:
        """Execute a shell command within the project workspace.

        Args:
            command: The shell command to execute.
            cwd: Optional relative working directory inside the workspace.
            timeout: Maximum execution time in seconds.
            confirmed: Set to true to proceed past a confirmation_required result.
        """
        return await asyncio.to_thread(
            _run_bash,
            command,
            active,
            cwd,
            timeout,
            confirmed,
        )

    return run_bash


# ── public factory ────────────────────────────────────────────────────────────


def make_builtin_tools(policy: SandboxPolicy | None = None) -> list[Callable]:
    """
    Return the standard built-in tool list, all bound to *policy*.

    Usage::

        from policy import SandboxPolicy, Permission

        tools = make_builtin_tools(
            SandboxPolicy(
                allow={Permission.READ, Permission.WRITE, Permission.EXECUTE},
                require_confirmation={Permission.DELETE, Permission.SYSTEM, Permission.INSTALL},
            )
        )
        config = AgentConfig(model=..., tools=tools)

    When *policy* is ``None`` the tools read ``SANDBOX_ALLOW`` /
    ``SANDBOX_CONFIRM`` from the environment at the first call.
    """
    return [
        read_file,
        write_file,
        _make_execute_python(policy),
        _make_run_bash(policy),
    ]


# ── legacy flat list (backwards-compatible) ───────────────────────────────────

#: Pre-built tool list using the env-var policy.
#: Equivalent to ``make_builtin_tools()``.
#: Prefer :func:`make_builtin_tools` for explicit policy control.
BUILTIN_TOOLS: list[Callable] = make_builtin_tools()
