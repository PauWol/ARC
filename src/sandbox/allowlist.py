from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from src.policy import SandboxPolicy

MAX_MEMORY_MB = 256
TIMEOUT_SECONDS = 3

SANDBOX_ROOT = Path("./sandbox_workspace").resolve()
SANDBOX_ROOT.mkdir(exist_ok=True)


def safe_open(path, mode="r", *args, **kwargs):
    full = (SANDBOX_ROOT / path).resolve()
    if not str(full).startswith(str(SANDBOX_ROOT)):
        raise PermissionError("Path escape blocked")
    return open(full, mode, *args, **kwargs)


#: Always available — pure value-manipulation, no I/O or introspection.
SAFE_BUILTINS: dict[str, Any] = {
    # output
    "print": print,
    # numerics
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    "chr": chr,
    "ord": ord,
    "complex": complex,
    # iterables
    "enumerate": enumerate,
    "sorted": sorted,
    "reversed": reversed,
    "zip": zip,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "next": next,
    "iter": iter,
    # constructors
    "list": list,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "tuple": tuple,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "slice": slice,
    # type / reflection (safe subset)
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    "hasattr": hasattr,
    "id": id,
    "hash": hash,
    "repr": repr,
    "format": format,
    # OOP helpers
    "object": object,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "super": super,
    # file I/O — sandboxed path
    "open": safe_open,
    # common exceptions
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "StopIteration": StopIteration,
    "RuntimeError": RuntimeError,
    "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError,
    "ZeroDivisionError": ZeroDivisionError,
    "IOError": IOError,
    "OSError": OSError,
    "FileNotFoundError": FileNotFoundError,
    "PermissionError": PermissionError,
    "ImportError": ImportError,
    "AssertionError": AssertionError,
}

#: Never permitted regardless of policy.
DANGEROUS: list[str] = [
    "eval",
    "exec",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "__import__",  # replaced by safe_import in runner
]

#: Attribute names that are always blocked in _safe_getattr_.
BLOCKED_ATTRS: frozenset[str] = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__dict__",
        "__module__",
        "__qualname__",
        "__builtins__",
        "__spec__",
        "__loader__",
        "__package__",
        "__path__",
        "__file__",
    }
)


# ── module registry ───────────────────────────────────────────────────────────

#: Available with any non-empty policy (no special permission needed).
_BASE_MODULES: frozenset[str] = frozenset(
    {
        "math",
        "cmath",
        "json",
        "statistics",
        "datetime",
        "random",
        "pathlib",
        "collections",
        "itertools",
        "re",
        "string",
        "textwrap",
        "functools",
        "operator",
        "copy",
        "pprint",
        "enum",
        "dataclasses",
        "typing",
        "abc",
        "contextlib",
        "decimal",
        "fractions",
        "heapq",
        "bisect",
        "array",
        "struct",
        "uuid",
        "hashlib",
        "base64",
        "binascii",
        "time",
        "calendar",
    }
)

#: Module root → Permission name required.
_GATED_MODULES: dict[str, str] = {
    # ── READ ──────────────────────────────────────────────────────────────────
    "glob": "READ",
    "fnmatch": "READ",
    "csv": "READ",
    "configparser": "READ",
    "tomllib": "READ",
    "xml": "READ",
    "html": "READ",
    "difflib": "READ",
    # ── WRITE ─────────────────────────────────────────────────────────────────
    "os": "WRITE",  # broad; dangerous attrs blocked by _safe_getattr_
    "io": "WRITE",
    "shutil": "WRITE",  # rmtree etc. still require DELETE at runtime
    "tempfile": "WRITE",
    "pickle": "WRITE",
    "shelve": "WRITE",
    "zipfile": "WRITE",
    "tarfile": "WRITE",
    "gzip": "WRITE",
    "bz2": "WRITE",
    "lzma": "WRITE",
    # ── EXECUTE ───────────────────────────────────────────────────────────────
    "subprocess": "EXECUTE",
    "threading": "EXECUTE",
    "multiprocessing": "EXECUTE",
    "asyncio": "EXECUTE",
    "concurrent": "EXECUTE",
    "signal": "EXECUTE",
    "pty": "EXECUTE",
    # ── NETWORK ───────────────────────────────────────────────────────────────
    "socket": "NETWORK",
    "ssl": "NETWORK",
    "urllib": "NETWORK",
    "http": "NETWORK",
    "email": "NETWORK",
    "smtplib": "NETWORK",
    "ftplib": "NETWORK",
    "imaplib": "NETWORK",
    "poplib": "NETWORK",
    "xmlrpc": "NETWORK",
    "requests": "NETWORK",
    "httpx": "NETWORK",
    "aiohttp": "NETWORK",
    "websockets": "NETWORK",
    # ── SYSTEM ────────────────────────────────────────────────────────────────
    "ctypes": "SYSTEM",
    "mmap": "SYSTEM",
    "platform": "SYSTEM",
    "resource": "SYSTEM",
    "pwd": "SYSTEM",
    "grp": "SYSTEM",
    "termios": "SYSTEM",
    "tty": "SYSTEM",
    # ── INSTALL ───────────────────────────────────────────────────────────────
    "pip": "INSTALL",
    "setuptools": "INSTALL",
    "pkg_resources": "INSTALL",
    "importlib": "INSTALL",
}


# ── AST node rules ────────────────────────────────────────────────────────────

#: Always forbidden — no policy unlocks these.
_ALWAYS_FORBIDDEN: tuple[type[ast.AST], ...] = (
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
)


# ── public API ────────────────────────────────────────────────────────────────


def build_python_profile(policy: "SandboxPolicy") -> dict:
    """
    Return a runtime profile derived from *policy*.

    Keys
    ----
    safe_modules : frozenset[str]
        Module roots the sandbox's ``safe_import`` will allow.
    forbidden_nodes : tuple[type[ast.AST], ...]
        AST node types that :class:`SecurityVisitor` will reject.
    allow_async : bool
        Whether ``async def`` / ``await`` are permitted.
    """
    from policy import Permission  # local import to avoid circular at module level

    # Build allowed module set
    modules: set[str] = set(_BASE_MODULES)
    for root, perm_name in _GATED_MODULES.items():
        perm = Permission[perm_name]
        if policy.is_allowed(perm) or policy.needs_confirmation(perm):
            # Unlock the import; actual runtime destructiveness (DELETE etc.)
            # is a separate concern for the code being executed.
            modules.add(root)

    # Build forbidden node list
    forbidden: list[type[ast.AST]] = list(_ALWAYS_FORBIDDEN)

    # ClassDef / Lambda: fine for medium-trust (dunder guards prevent abuse)
    if not policy.has_any():
        forbidden.extend([ast.ClassDef, ast.Lambda])

    # AsyncFunctionDef: only if EXECUTE is in play
    if not (
        policy.is_allowed(Permission.EXECUTE)
        or policy.needs_confirmation(Permission.EXECUTE)
    ):
        forbidden.append(ast.AsyncFunctionDef)

    return {
        "safe_modules": frozenset(modules),
        "forbidden_nodes": tuple(forbidden),
        "allow_async": (
            policy.is_allowed(Permission.EXECUTE)
            or policy.needs_confirmation(Permission.EXECUTE)
        ),
    }
