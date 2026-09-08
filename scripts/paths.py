"""Resolve project data paths as POSIX paths relative to the repo root."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DOTENV_LOADED = False

_DEFAULT_PANEL = "data/processed/wem_5min_panel.parquet"
_DEFAULT_RAW = "raw"


def load_env() -> None:
    """Load `.env` from the project root if present. Existing env vars win."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_simple(env_path)
        return
    load_dotenv(env_path, override=False)


def _load_env_simple(env_path: Path) -> None:
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def as_posix_path(value: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    """Interpret `value` as a POSIX path (forward slashes).

    Relative paths are resolved from the project root (or `base`).
    """
    text = os.fspath(value).replace("\\", "/").strip()
    if not text:
        raise ValueError("path is empty")
    if text.startswith("/"):
        return Path(text)
    parts = [part for part in text.split("/") if part and part != "."]
    return (base or PROJECT_ROOT).joinpath(*parts)


def panel_path() -> Path:
    load_env()
    return as_posix_path(os.environ.get("PANEL_PATH", _DEFAULT_PANEL))


def raw_data_dir() -> Path:
    load_env()
    return as_posix_path(os.environ.get("RAW_DATA_DIR", _DEFAULT_RAW))
