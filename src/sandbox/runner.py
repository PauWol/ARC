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

from src.registry import ToolResult

from src.sandbox.allowlist import (
    SAFE_BUILTINS,
    DANGEROUS,
    FORBIDDEN_NODES,
    MAX_MEMORY_MB,
    SAFE_MODULES,
    TIMEOUT_SECONDS,
    BLOCKED_ATTRS,
)


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in SAFE_MODULES:
        raise ImportError(f"Blocked import: {name}")
    return __import__(name, globals, locals, fromlist, level)


_MERGED_BUILTINS: dict = {**safe_builtins, **SAFE_BUILTINS}
for _name in DANGEROUS:
    _MERGED_BUILTINS.pop(_name, None)

_MERGED_BUILTINS["__import__"] = safe_import


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


def _safe_getattr_(obj, name: str, *args):
    if name.startswith("_"):
        raise AttributeError(f"[Sandbox] Access to '{name}' is not allowed")

    if name in BLOCKED_ATTRS:
        raise AttributeError(f"[Sandbox] Access to '{name}' is not allowed")

    return getattr(obj, name, *args)


class SecurityVisitor(ast.NodeVisitor):
    def visit_Name(self, node: ast.Name):
        if "__" in node.id:
            raise ValueError(f"Dunder access forbidden: {node.id}")
        self.generic_visit(node)  # BUG FIX: was missing — child nodes skipped

    def visit_Attribute(self, node: ast.Attribute):
        if "__" in node.attr:
            raise ValueError(f"Dunder attribute forbidden: {node.attr}")
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"Forbidden syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in SAFE_MODULES:
                raise ValueError(f"Import blocked: {root}")

    def visit_ImportFrom(self, node):
        root = (node.module or "").split(".")[0]
        if root not in SAFE_MODULES:
            raise ValueError(f"Import blocked: {root}")


def _validate_code(code: str) -> None:
    tree = ast.parse(code)
    SecurityVisitor().visit(tree)


def _worker(code: str, conn: Connection) -> None:
    try:
        # ── 1. Resource limits ────────────────────────────────────────────────
        memory_bytes = MAX_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_SECONDS, TIMEOUT_SECONDS))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        # ── 2. AST validation ─────────────────────────────────────────────────
        _validate_code(code)

        # ── 3. Compile ────────────────────────────────────────────────────────
        # BUG FIX: suppress RestrictedPython's SyntaxWarning about 'printed'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            byte_code = compile(code, filename="<sandbox>", mode="exec")

        buffer = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buffer

        def _print_(*args, **kwargs):
            print(*args, file=buffer, **kwargs)

        # ── 4. Build sandbox globals ──────────────────────────────────────────
        globals_dict: dict = {
            "__builtins__": dict(_MERGED_BUILTINS),
            # print — PrintCollector class; instance lands in globals as '_print'
            "_print_": _print_,
            # iteration
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            # item + attribute access
            "_getitem_": default_guarded_getitem,
            "_getattr_": _safe_getattr_,  # BUG FIX: no raw getattr
            # augmented assignment
            "_inplacevar_": _inplacevar_,  # BUG FIX: full operator set
            # write guard
            "_write_": lambda x: x,
        }

        # add common modules as already imported
        globals_dict.update(
            {
                "math": math,
                "json": json,
                "re": re,
                "statistics": statistics,
            }
        )

        # ── 5. Execute ────────────────────────────────────────────────────────
        locals_dict: dict = {}
        exec(byte_code, globals_dict, locals_dict)  # noqa: S102

        sys.stdout = sys_stdout
        output = buffer.getvalue()

        conn.send({"success": True, "output": output})

    except Exception:
        conn.send({"success": False, "error": traceback.format_exc(limit=3)})

    finally:
        conn.close()


def run_python(code: str) -> ToolResult:
    """
    Execute `code` in a sandboxed child process.
    """
    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(target=_worker, args=(code, child_conn))
    process.start()

    child_conn.close()

    process.join(TIMEOUT_SECONDS)

    if process.is_alive():
        process.kill()
        process.join()
        return ToolResult(
            False, "running python code failed", {"error": "Execution timed out"}
        )

    if not parent_conn.poll():
        return ToolResult(
            False, "running python code failed", {"error": "No output returned"}
        )

    result = parent_conn.recv()
    parent_conn.close()

    if result in ("error", "Traceback"):
        return ToolResult(False, "running python code failed", {"result": result})

    return ToolResult(True, "ran python code successfully", {"result": result})
