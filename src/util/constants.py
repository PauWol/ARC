from enum import Enum
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


class CONST(Enum):
    AGENT_WORKSPACE = "AGENT_WORKSPACE"


DEFAULT_DOT_ENV = {CONST.AGENT_WORKSPACE.value: "~/arc/workspace"}


def env(key: str, missing_error: bool = True) -> Any:
    _v = os.getenv(key)

    if missing_error and not _v:
        raise EnvironmentError(f"Missing '.env' entry {key} is {_v}")

    return _v


def workspcae_path(path: str | None = None) -> Path:
    """Return the Agent-Workspace-Path if a path is provided the full resolved is returned."""

    _w_space = Path(env(CONST.AGENT_WORKSPACE.value)).expanduser().as_posix().resolve()
    _w_space.mkdir(parents=True, exist_ok=True)

    full = _w_space

    if path:
        full = (_w_space / path).resolve()

        if not str(full).startswith(str(_w_space)):
            raise Exception("Path escape blocked")

    return full
