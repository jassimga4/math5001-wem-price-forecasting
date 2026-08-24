#!/usr/bin/env python3
"""Print a summary of the processed 5-minute WEM modelling panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PANEL = Path("data/processed/wem_5min_panel.parquet")


def main() -> None:
    if not PANEL.exists():
        raise SystemExit(f"missing {PANEL}")

    df = pd.read_parquet(PANEL)
    print(f"file: {PANEL}")
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
