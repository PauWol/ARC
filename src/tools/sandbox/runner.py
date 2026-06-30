from __future__ import annotations

import ast
from io import StringIO
import math
import json
import re
import statistics
import multiprocessing as mp
from multiprocessing.connection import Connection
import resource
import sys
import traceback
import warnings

from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem

from src.schema import ToolResult

from src.tools.sandbox.allowlist import (
    SAFE_BUILTINS,
    DANGEROUS,
    MAX_MEMORY_MB,
    TIMEOUT_SECONDS,
    BLOCKED_ATTRS,
    build_python_profile,
)
from src.tools.sandbox.policy import SandboxPolicy, DEFAULT_POLICY


# ── safe import factory ───────────────────────────────────────────────────────


def _make_safe_import(safe_modules: frozenset[str]):
    """Return an ``__import__`` replacement that enforces *safe_modules*."""

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in safe_modules:
            raise ImportError(f"Import blocked by sandbox policy: {name!r}")
        return __import__(name, globals, locals, fromlist, level)

    return safe_import


# ── merged builtins factory ───────────────────────────────────────────────────


def _make_builtins(safe_import_fn) -> dict:
    merged = {**safe_builtins, **SAFE_BUILTINS}
    for name in DANGEROUS:
        merged.pop(name, None)
    merged["__import__"] = safe_import_fn
    return merged


# ── augmented assignment helper ───────────────────────────────────────────────


def _inplacevar_(op: str, x, y):
    match op:
        case "+=":
            return x + y
        case "-=":
            return x - y
        case "*=":
            return x * y
        case "/=":
            return x / y
        case "//=":
            return x // y
        case "%=":
            return x % y
        case "**=":
            return x**y
        case "&=":
            return x & y
        case "|=":
            return x | y
        case "^=":
            return x ^ y
        case ">>=":
            return x >> y
        case "<<=":
            return x << y
        case _:
            raise ValueError(f"[Sandbox] Unsupported inplace operator: {op}")


# ── attribute guard ───────────────────────────────────────────────────────────


def _safe_getattr_(obj, name: str, *args):
    if name.startswith("_"):
        raise AttributeError(f"[Sandbox] Access to {name!r} is not allowed")
    if name in BLOCKED_ATTRS:
        raise AttributeError(f"[Sandbox] Access to {name!r} is not allowed")
    return getattr(obj, name, *args)


# ── AST security visitor ──────────────────────────────────────────────────────


class SecurityVisitor(ast.NodeVisitor):
    """Walk the AST and raise on any construct forbidden by *forbidden_nodes*."""

    def __init__(self, forbidden_nodes: tuple[type[ast.AST], ...]) -> None:
        self._forbidden = forbidden_nodes

    def visit_Name(self, node: ast.Name):
        if "__" in node.id:
            raise ValueError(f"Dunder name access forbidden: {node.id!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if "__" in node.attr:
            raise ValueError(f"Dunder attribute forbidden: {node.attr!r}")
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST):
        if isinstance(node, self._forbidden):
            raise ValueError(
                f"Forbidden syntax ({type(node).__name__}) is not permitted "
                f"by the current sandbox policy."
            )
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import):
        # Deferred to safe_import at runtime; visit children for dunder checks.
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.generic_visit(node)


def _validate_code(code: str, forbidden_nodes: tuple[type[ast.AST], ...]) -> None:
    tree = ast.parse(code)
    SecurityVisitor(forbidden_nodes).visit(tree)


# ── worker (runs in child process) ───────────────────────────────────────────


def _worker(code: str, conn: Connection, profile: dict) -> None:
    try:
        # Resource limits
        memory_bytes = MAX_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_SECONDS, TIMEOUT_SECONDS))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        # Static analysis
        _validate_code(code, profile["forbidden_nodes"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            byte_code = compile(code, filename="<sandbox>", mode="exec")

        # I/O capture
        buffer = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buffer

        def _print_(*args, **kwargs):
            print(*args, file=buffer, **kwargs)

        safe_import = _make_safe_import(profile["safe_modules"])
        builtins = _make_builtins(safe_import)

        globals_dict: dict = {
            "__builtins__": builtins,
            # print shim
            "_print_": _print_,
            # iteration
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            # item + attribute access
            "_getitem_": default_guarded_getitem,
            "_getattr_": _safe_getattr_,
            # augmented assignment
            "_inplacevar_": _inplacevar_,
            # write guard
            "_write_": lambda x: x,
            # pre-imported safe modules
            "math": math,
            "json": json,
            "re": re,
            "statistics": statistics,
        }

        exec(byte_code, globals_dict, {})  # noqa: S102

        sys.stdout = sys_stdout
        conn.send({"success": True, "output": buffer.getvalue()})

    except Exception:
        conn.send({"success": False, "error": traceback.format_exc(limit=3)})

    finally:
        conn.close()


# ── public entry point ────────────────────────────────────────────────────────


def run_python(code: str, policy: SandboxPolicy | None = None) -> ToolResult:
    """
    Execute *code* in a sandboxed child process subject to *policy*.

    Parameters
    ----------
    code:
        Python source to execute.
    policy:
        :class:`~policy.SandboxPolicy` controlling which imports and
        syntax constructs are permitted.  Defaults to
        :data:`~policy.DEFAULT_POLICY`.
    """
    if policy is None:
        policy = DEFAULT_POLICY

    profile = build_python_profile(policy)

    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(target=_worker, args=(code, child_conn, profile))
    process.start()
    child_conn.close()

    process.join(TIMEOUT_SECONDS)

    if process.is_alive():
        process.kill()
        process.join()
        return ToolResult(
            False,
            "running python code failed",
            {"error": "Execution timed out"},
        )

    if not parent_conn.poll():
        return ToolResult(
            False,
            "running python code failed",
            {"error": "No output returned from worker"},
        )

    result = parent_conn.recv()
    parent_conn.close()

    if not isinstance(result, dict):
        return ToolResult(False, "running python code failed", {"result": result})

    if result.get("success"):
        return ToolResult(
            True,
            "ran python code successfully",
            {"result": result},
        )
    else:
        return ToolResult(
            False,
            "running python code failed",
            {"result": result},
        )
