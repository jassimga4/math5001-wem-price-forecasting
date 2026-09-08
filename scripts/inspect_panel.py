#!/usr/bin/env python3
"""Print a summary of the processed 5-minute WEM modelling panel."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.paths import panel_path


def _require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Parquet support is missing in this Python environment.\n"
            "Use the project virtualenv (includes pyarrow):\n"
            "  source .venv/bin/activate\n"
            "  python scripts/inspect_panel.py\n"
            "Or:  python -m pip install pyarrow"
        ) from exc


def main() -> None:
    panel = panel_path()
    if not panel.exists():
        raise SystemExit(f"missing {panel.as_posix()}")

    _require_parquet_engine()
    df = pd.read_parquet(panel, engine="pyarrow")
    print(f"file: {panel.as_posix()}")
    print(f"rows: {len(df):,}")
    print(f"cols: {len(df.columns)}")
    print("columns:")
    for name in df.columns:
        print(f"  - {name}")
    if "interval_end" in df.columns:
        ts = pd.to_datetime(df["interval_end"])
        print(f"range: {ts.min()} → {ts.max()}")
    print()
    print(df.head(5).to_string(index=False))
    print()
    print("null share:")
    nulls = df.isna().mean().sort_values(ascending=False)
    for name, share in nulls.items():
        print(f"  {name:28s} {100 * share:6.2f}%")


if __name__ == "__main__":
    main()
