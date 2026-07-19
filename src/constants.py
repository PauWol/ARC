import os
from pathlib import Path

from dotenv import load_dotenv
from typing import TypeVar

load_dotenv(Path(".env"), override=False)

T = TypeVar("T")


def get_env(key: str, default: T | None = None) -> str | T | None:
    """
    Return an environment variable or a default value.
    """
    return os.getenv(key, default)


def get_env_keys(env_file: str | Path = ".env") -> dict[str, str]:
    """
    Load a .env file (if it exists) and return all environment variables.

    Args:
        env_file: Path to the .env file.

    Returns:
        Dictionary of environment variable names and values.
    """
    env_path = Path(env_file)

    if env_path.exists():
        load_dotenv(env_path, override=False)

    return dict(os.environ)


def path(p: str | Path) -> Path:
    """Helper Path function to expand all '~' string paths to full."""
    return Path(p).expanduser()


ENV_PATH = path("~/arc/.env")

DEFAULT_DOT_ENV = {
    "ARC_DIR": "~/arc",
    "AGENT_WORKSPACE": "~/arc/workspace",
    "LLM_MODEL_STORE": "~/arc/models",
    "HF_TOKEN": "YOUR-HUGGINGFACE-TOKEN-OPTIONAL",
    "LOG_LEVEL": "INFO",
    "LOG_FILE": "~/arc/workspace/agent.log",
    "LOG_CONSOLE": True,
    "LOG_JSON": False,
    "LOG_ROTATE": True,
    "LOG_MAX_BYTES": 10485760,
    "LOG_BACKUP_COUNT": 2,
    "SANDBOX_ALLOW": "READ,WRITE,EXECUTE,NETWORK",
    "SANDBOX_CONFIRM": "DELETE,SYSTEM,INSTALL",
}

_DEV = DEFAULT_DOT_ENV

# Project Directories
ARC_DIR = path(get_env("ARC_DIR", _DEV["ARC_DIR"]))  # pyright: ignore[reportArgumentType]
AGENT_WORKSPACE = path(get_env("AGENT_WORKSPACE", _DEV["AGENT_WORKSPACE"]))  # pyright: ignore[reportArgumentType]

# LLM Model Management
LLM_MODEL_STORE = path(get_env("LLM_MODEL_STORE", _DEV["LLM_MODEL_STORE"]))  # pyright: ignore[reportArgumentType]
HF_TOKEN = get_env("HF_TOKEN")


# Logging
LOG_LEVEL = get_env("LOG_LEVEL", _DEV["LOG_LEVEL"])
LOG_FILE = path(get_env("LOG_FILE", _DEV["LOG_FILE"]))  # pyright: ignore[reportArgumentType]
LOG_CONSOLE = get_env("LOG_CONSOLE", _DEV["LOG_CONSOLE"])
LOG_JSON = get_env("LOG_JSON", _DEV["LOG_JSON"])
LOG_ROTATE = get_env("LOG_ROTATE", _DEV["LOG_ROTATE"])
LOG_MAX_BYTES = get_env("LOG_MAX_BYTES", _DEV["LOG_MAX_BYTES"])
LOG_BACKUP_COUNT = get_env("LOG_BACKUP_COUNT", _DEV["LOG_BACKUP_COUNT"])


# Permissions
SANDBOX_ALLOW = get_env("SANDBOX_ALLOW", _DEV["SANDBOX_ALLOW"])
SANDBOX_CONFIRM = get_env("SANDBOX_CONFIRM", _DEV["SANDBOX_CONFIRM"])


def workspcae_path(_path: str | None = None) -> Path:
    """Return the Agent-Workspace-Path if a path is provided the full resolved is returned."""

    _w_space = path(AGENT_WORKSPACE)  # pyright: ignore[reportArgumentType]
    _w_space.mkdir(parents=True, exist_ok=True)

    full = _w_space

    if path:
        full = (_w_space / Path(_path)).resolve()  # pyright: ignore[reportArgumentType]

        if not str(full).startswith(str(_w_space)):
            raise Exception("Path escape blocked")

    return full
