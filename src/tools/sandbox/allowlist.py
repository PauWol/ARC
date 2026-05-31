import ast
from pathlib import Path

MAX_MEMORY_MB = 256
TIMEOUT_SECONDS = 3

SANDBOX_ROOT = Path("./sandbox_workspace").resolve()
SANDBOX_ROOT.mkdir(exist_ok=True)


def safe_open(path, mode="r", *args, **kwargs):
    full = (SANDBOX_ROOT / path).resolve()

    if not str(full).startswith(str(SANDBOX_ROOT)):
        raise PermissionError("Path escape blocked")

    return open(full, mode, *args, **kwargs)


SAFE_BUILTINS = {
    "print": print,
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "enumerate": enumerate,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "open": safe_open,
}


# Remove dangerous builtins
DANGEROUS = [
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
]


FORBIDDEN_NODES = (
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
)

SAFE_MODULES = {
    "math",
    "json",
    "statistics",
    "datetime",
    "random",
    "pathlib",
    "collections",
    "itertools",
    "re",
}

BLOCKED_ATTRS = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
}
