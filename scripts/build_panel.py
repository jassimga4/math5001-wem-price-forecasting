#!/usr/bin/env python3
"""Build a 5-minute WEM modelling panel from downloaded AEMO CSVs.

Spine: post-reform Market Clearing Price (energy).
Joined:
  - operational demand / withdrawal (5 min)
  - estimated distributed PV (5 min)
  - reference trading price (30 min, mapped onto the 6 dispatch intervals)
  - STEM clearing price / quantity (30 min)
  - sent-out generation and legacy 30-min DPV
  - facility SCADA summed to system MWh / MW
  - market schedule energy summed to system MW

Usage (in Docker):
    docker compose run --rm shell python scripts/build_panel.py --data-dir "/Users/jg/Desktop/Demand Forecasting AEMO/data"
    docker compose run --rm shell python scripts/build_panel.py --skip-scada
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("build_panel")

REFORM_START = pd.Timestamp("2023-10-01 08:00:00")
DMY = "%d/%m/%Y %H:%M:%S"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed/wem_5min_panel.parquet"))
    p.add_argument("--start", default="2023-10-01 08:00:00")
    p.add_argument("--end", default=None)
    p.add_argument("--skip-scada", action="store_true")
    p.add_argument("--skip-schedule", action="store_true")
    return p.parse_args()


def _read_csvs(paths: list[str], **kwargs) -> pd.DataFrame:
    frames = []
    for path in paths:
        log.info("reading %s", path)
        frames.append(pd.read_csv(path, **kwargs))
    if not frames:
        raise FileNotFoundError(f"no files matched: {paths}")
    return pd.concat(frames, ignore_index=True)


def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    """Parse AEMO timestamps. Post-reform price files are DD/MM/YYYY; older files are ISO."""
    raw = series.astype(str).str.strip().str.strip('"')
    parsed = pd.to_datetime(raw, format=DMY, errors="coerce")
    missing = parsed.isna() & raw.ne("") & raw.ne("nan")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw.loc[missing], errors="coerce")
    return parsed


def trading_interval_end(ts: pd.Series) -> pd.Series:
    """Map a 5-minute dispatch interval-end onto its 30-minute trading interval-end."""
    ts = pd.to_datetime(ts)
    minute = ts.dt.minute
    hour_start = ts.dt.floor("h")
    out = hour_start + pd.Timedelta(minutes=30)
    out = out.mask(minute.eq(0), hour_start)
    out = out.mask(minute.gt(30), hour_start + pd.Timedelta(hours=1))
    return out


def load_mcp(data_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data_dir / "market-clearing-prices" / "MarketClearingPrices-*.csv")))
    df = _read_csvs(paths)
    df["interval_end"] = parse_mixed_datetime(df["Dispatch Interval"])
    out = pd.DataFrame(
        {
            "interval_end": df["interval_end"],
            "mcp": pd.to_numeric(df["Energy Clearing Price"], errors="coerce"),
            "cr_raise": pd.to_numeric(df["Contingency Raise Clearing Price"], errors="coerce"),
            "cr_lower": pd.to_numeric(df["Contingency Lower Clearing Price"], errors="coerce"),
            "reg_raise": pd.to_numeric(df["Regulation Raise Clearing Price"], errors="coerce"),
            "reg_lower": pd.to_numeric(df["Regulation Lower Clearing Price"], errors="coerce"),
            "rocof": pd.to_numeric(df["RoCoF Clearing Price"], errors="coerce"),
        }
    )
    return (
        out.dropna(subset=["interval_end"])
        .sort_values("interval_end")
        .drop_duplicates("interval_end", keep="last")
        .reset_index(drop=True)
    )


def load_withdrawal(data_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data_dir / "operational-demand-withdrawal" / "OperationalDemandWithdrawal-*.csv")))
    df = _read_csvs(paths)
    return pd.DataFrame(
        {
            "interval_end": parse_mixed_datetime(df["Dispatch Interval"]),
            "operational_demand_mw": pd.to_numeric(df["Operational Demand"], errors="coerce"),
            "unscheduled_demand_mw": pd.to_numeric(df["Unscheduled Operational Demand"], errors="coerce"),
            "operational_withdrawal_mw": pd.to_numeric(df["Operational Withdrawal"], errors="coerce"),
        }
    ).dropna(subset=["interval_end"]).drop_duplicates("interval_end", keep="last")


def load_estimated_dpv(data_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data_dir / "estimated-dpv" / "distributed-pv-new-*.csv")))
    df = _read_csvs(paths)
    return pd.DataFrame(
        {
            "interval_end": parse_mixed_datetime(df["Timestamp"]),
            "dpv_mw": pd.to_numeric(df["Estimated DPV Generation(MW)"], errors="coerce"),
        }
    ).dropna(subset=["interval_end"]).drop_duplicates("interval_end", keep="last")


def load_rtp(data_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data_dir / "reference-trading-price" / "ReferenceTradingPrice-*.csv")))
    df = _read_csvs(paths)
    return pd.DataFrame(
        {
            "trading_interval_end": parse_mixed_datetime(df["Trading Interval"]),
            "rtp": pd.to_numeric(df["Reference Trading Price"], errors="coerce"),
        }
    ).dropna(subset=["trading_interval_end"]).drop_duplicates("trading_interval_end", keep="last")


def load_stem(data_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data_dir / "stem-summary" / "stem-summary-*.csv")))
    df = _read_csvs(paths)
    return pd.DataFrame(
        {
            "trading_interval_end": parse_mixed_datetime(df["Trading Interval"]),
            "stem_price": pd.to_numeric(df["Clearing Price ($/MWh)"], errors="coerce"),
            "stem_qty_mwh": pd.to_numeric(df["Clearing Quantity (MWh)"], errors="coerce"),
            "stem_bid_mwh": pd.to_numeric(df["Bid Quantity (MWh)"], errors="coerce"),
            "stem_offer_mwh": pd.to_numeric(df["Offer Quantity (MWh)"], errors="coerce"),
        }
    ).dropna(subset=["trading_interval_end"]).drop_duplicates("trading_interval_end", keep="last")


def load_tt30gen(data_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data_dir / "tt30gen" / "total-sent-out-generation-*.csv")))
    df = _read_csvs(paths)
    mwh = pd.to_numeric(df["Total Sent Out Generation (MWh)"], errors="coerce")
    return pd.DataFrame(
        {
            "trading_interval_end": parse_mixed_datetime(df["Trading Interval"]),
            "sent_out_mwh": mwh,
            "sent_out_mw": mwh * 2.0,
        }
    ).dropna(subset=["trading_interval_end"]).drop_duplicates("trading_interval_end", keep="last")


def load_dpv_30min(data_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data_dir / "distributed-pv" / "distributed-pv-*.csv")))
    df = _read_csvs(paths)
    return pd.DataFrame(
        {
            "trading_interval_end": parse_mixed_datetime(df["Trading Interval"]),
            "dpv_mw_30min": pd.to_numeric(df["Estimated DPV Generation (MW)"], errors="coerce"),
        }
    ).dropna(subset=["trading_interval_end"]).drop_duplicates("trading_interval_end", keep="last")


def _sum_monthly(paths: list[str], time_col: str, value_col: str, out_name: str) -> pd.DataFrame:
    parts = []
    for path in paths:
        log.info("aggregating %s", path)
        chunk = pd.read_csv(path, usecols=[time_col, value_col])
        chunk[time_col] = parse_mixed_datetime(chunk[time_col])
        chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
        g = (
            chunk.dropna(subset=[time_col])
            .groupby(time_col, sort=False)[value_col]
            .sum()
            .rename(out_name)
            .reset_index()
            .rename(columns={time_col: "interval_end"})
        )
        parts.append(g)
        del chunk
    if not parts:
        return pd.DataFrame(columns=["interval_end", out_name])
    return (
        pd.concat(parts, ignore_index=True)
        .groupby("interval_end", as_index=False)[out_name]
        .sum()
    )


def load_scada(data_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data_dir / "facility-scada" / "FacilityScada-*.csv")))
    df = _sum_monthly(paths, "Dispatch Interval", "Average MWh", "scada_mwh")
    df["scada_mw"] = df["scada_mwh"] * 12.0
    return df


def load_schedule(data_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data_dir / "market-schedule" / "MarketSchedule-*.csv")))
    return _sum_monthly(paths, "Dispatch Interval", "Energy Schedule", "scheduled_energy_mw")


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["interval_end"]
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["year"] = ts.dt.year
    df["is_weekend"] = df["dow"] >= 5
    return df


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else None

    panel = load_mcp(args.data_dir)
    log.info("MCP rows %s  %s → %s", len(panel), panel["interval_end"].min(), panel["interval_end"].max())

    panel = panel.merge(load_withdrawal(args.data_dir), on="interval_end", how="left")
    panel = panel.merge(load_estimated_dpv(args.data_dir), on="interval_end", how="left")

    panel["trading_interval_end"] = trading_interval_end(panel["interval_end"])
    panel = panel.merge(load_rtp(args.data_dir), on="trading_interval_end", how="left")
    panel = panel.merge(load_stem(args.data_dir), on="trading_interval_end", how="left")
    panel = panel.merge(load_tt30gen(args.data_dir), on="trading_interval_end", how="left")
    panel = panel.merge(load_dpv_30min(args.data_dir), on="trading_interval_end", how="left")

    if not args.skip_scada:
        panel = panel.merge(load_scada(args.data_dir), on="interval_end", how="left")
    if not args.skip_schedule:
        panel = panel.merge(load_schedule(args.data_dir), on="interval_end", how="left")

    panel = panel.loc[panel["interval_end"] >= start]
    if end is not None:
        panel = panel.loc[panel["interval_end"] <= end]
    panel = panel.sort_values("interval_end").reset_index(drop=True)

    panel["residual_demand_mw"] = panel["operational_demand_mw"] - panel["dpv_mw"]
    panel["mcp_lag1"] = panel["mcp"].shift(1)
    panel["mcp_lag6"] = panel["mcp"].shift(6)
    panel["mcp_lag12"] = panel["mcp"].shift(12)
    panel = add_calendar(panel)

    expected = pd.date_range(panel["interval_end"].min(), panel["interval_end"].max(), freq="5min")
    missing = expected.difference(panel["interval_end"])
    log.info(
        "panel %s rows  %s → %s  missing 5-min slots %s",
        f"{len(panel):,}",
        panel["interval_end"].min(),
        panel["interval_end"].max(),
        f"{len(missing):,}",
    )
    coverage = {
        col: float(panel[col].notna().mean())
        for col in [
            "mcp",
            "operational_demand_mw",
            "dpv_mw",
            "rtp",
            "stem_price",
            "sent_out_mw",
            "scada_mw",
            "scheduled_energy_mw",
        ]
        if col in panel.columns
    }
    for col, share in coverage.items():
        log.info("coverage %-24s %6.1f%%", col, 100 * share)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)
    sample = args.out.with_name(args.out.stem + "_sample.csv")
    panel.head(200).to_csv(sample, index=False)
    log.info("wrote %s  (%.1f MB)", args.out, args.out.stat().st_size / 1e6)
    log.info("wrote %s", sample)


if __name__ == "__main__":
    main()
