from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from src.util.constants import env, CONST


api = HfApi()


class QUANTS(Enum):
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"


def str_to_quant(value: str) -> QUANTS | None:
    value = value.lower()
    for q in QUANTS:
        if q.value.lower() in value:
            return q
    return None


def get_model_store_dir() -> Path:
    path = Path(env(CONST.LLM_MODEL_STORE.value)).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_repo_dir(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _repo_dir_to_id(name: str) -> str:
    return name.replace("__", "/")


@dataclass(slots=True)
class Model:
    repo_id: str
    files: list[str] = field(default_factory=list)
    downloads: int = 0
    local_path: Path | None = None


def list_models() -> list[Model]:
    """
    List locally downloaded GGUF models from your model store.
    Layout expected:
        LLM_MODEL_STORE/
            repo__name/
                model.gguf
    """
    root = get_model_store_dir()
    models: list[Model] = []

    if not root.exists():
        return models

    for repo_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        gguf_files = sorted(repo_dir.rglob("*.gguf"))
        if not gguf_files:
            continue

        models.append(
            Model(
                repo_id=_repo_dir_to_id(repo_dir.name),
                files=[str(p.relative_to(repo_dir)) for p in gguf_files],
                local_path=repo_dir,
            )
        )

    return models


def search_gguf_models(query: str, limit: int = 20) -> list[Model]:
    """
    Search Hugging Face for models whose repo matches the query and that
    actually contain .gguf files.
    """
    results: list[Model] = []

    for model in api.list_models(
        search=query,
        sort="downloads",
        direction=-1,
        limit=limit,
    ):
        try:
            info = api.model_info(model.id)
        except Exception:
            continue

        gguf_files = [
            s.rfilename for s in info.siblings if s.rfilename.endswith(".gguf")
        ]
        if not gguf_files:
            continue

        results.append(
            Model(
                repo_id=model.id,
                downloads=getattr(model, "downloads", 0) or 0,
                files=gguf_files,
            )
        )

    return results


def download_gguf(
    repo_id: str,
    quant: QUANTS | str,
    out_dir: str | Path | None = None,
) -> Path:
    """
    Download all GGUF files matching a quantization pattern into a repo-specific folder.
    Example:
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF"
        quant=QUANTS.Q4_K_M
    """
    if isinstance(quant, str):
        quant_enum = str_to_quant(quant)
        if quant_enum is None:
            raise ValueError(f"Unknown quant: {quant}")
        quant = quant_enum

    store = Path(out_dir).expanduser().resolve() if out_dir else get_model_store_dir()
    target_dir = store / _safe_repo_dir(repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        allow_patterns=[f"*{quant.value}*.gguf"],
    )

    return target_dir


def download_gguf_file(
    repo_id: str,
    filename: str,
    out_dir: str | Path | None = None,
) -> Path:
    """
    Download one exact GGUF file into a repo-specific folder.
    """
    store = Path(out_dir).expanduser().resolve() if out_dir else get_model_store_dir()
    target_dir = store / _safe_repo_dir(repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
    )
    return Path(path)
