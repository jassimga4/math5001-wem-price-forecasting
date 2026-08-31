#!/usr/bin/env python3
"""Seaborn EDA on the full WEM 5-minute modelling panel.

Loads data/processed/wem_5min_panel.parquet, writes figures and a
per-column feature-interpretation report under reports/eda/.

Usage:
    cd "/Users/jg/Documents/MATH5001 Project"
    source .venv/bin/activate
    python scripts/explore_panel.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "wem_5min_panel.parquet"
OUT = ROOT / "reports" / "eda"
FIG = OUT / "figures"

TARGET = "mcp"
DT_COLS = ("interval_end", "trading_interval_end")
ID_LIKE = DT_COLS

# Domain notes used in the written interpretation. Stats are filled at runtime.
COLUMN_DOCS: dict[str, dict[str, str]] = {
    "interval_end": {
        "role": "index / time spine",
        "meaning": "End of the 5-minute dispatch interval (WEM energy MCP timestamp).",
        "use": "Not a regressor. Use for time-based CV, lags, and calendar features.",
        "watch": "Any contemporaneous feature joined on this stamp must match the forecast horizon or it is leakage.",
    },
    "mcp": {
        "role": "target",
        "meaning": "Energy market clearing price ($/MWh) for the 5-minute dispatch interval.",
        "use": "Primary modelling target for probabilistic / conformal price forecasts.",
        "watch": "Heavy tails and cap/floor spikes dominate RMSE. Prefer pinball, CRPS, coverage, and Spearman over raw Pearson.",
    },
    "cr_raise": {
        "role": "ancillary (FCAS) price",
        "meaning": "Contingency raise clearing price — paid to providers that can lift generation or cut load after a contingency.",
        "use": "Scarcity co-movement with energy; useful as a lag, not as a same-interval predictor unless FCAS is already known.",
        "watch": "Different market from energy. Spikes can be large while energy is moderate (and the reverse).",
    },
    "cr_lower": {
        "role": "ancillary (FCAS) price",
        "meaning": "Contingency lower clearing price — paid for the ability to reduce generation after a contingency.",
        "use": "Often more relevant in high-DPV, low-net-demand intervals.",
        "watch": "Zero-inflated / low median with occasional spikes. Log or rank transforms help.",
    },
    "reg_raise": {
        "role": "ancillary (FCAS) price",
        "meaning": "Regulation raise clearing price (secondary frequency control, raise).",
        "use": "Lagged regulation prices can mark tight control-reserve conditions.",
        "watch": "Collinear with other raise services during system stress; do not treat as an independent energy driver.",
    },
    "reg_lower": {
        "role": "ancillary (FCAS) price",
        "meaning": "Regulation lower clearing price (secondary frequency control, lower).",
        "use": "May rise when DPV ramps force downward regulation.",
        "watch": "Same-interval use is leakage for a price-nowcast of energy unless the service is already published.",
    },
    "rocof": {
        "role": "ancillary (FCAS) price",
        "meaning": "RoCoF (rate-of-change-of-frequency) control service clearing price.",
        "use": "Mostly a flag of inertia/contingency conditions; weak energy-price signal if mostly zero.",
        "watch": "Likely zero-inflated. Check the mass at zero before treating it as continuous.",
    },
    "operational_demand_mw": {
        "role": "fundamental (demand)",
        "meaning": "Operational demand (MW) on the SWIS at the dispatch interval.",
        "use": "Core load driver. Stronger as a lagged or forecasted input than as a same-interval observed value, depending on horizon.",
        "watch": "Highly collinear with unscheduled demand, SCADA, scheduled energy, and sent-out generation.",
    },
    "unscheduled_demand_mw": {
        "role": "fundamental (demand)",
        "meaning": "Unscheduled operational demand (MW). Near-duplicate of operational demand.",
        "use": "Drop or keep only one of {operational_demand_mw, unscheduled_demand_mw}.",
        "watch": "Pearson with operational_demand_mw will be ~1. Including both inflates VIF.",
    },
    "operational_withdrawal_mw": {
        "role": "fundamental (demand)",
        "meaning": "Operational withdrawal (MW). Small relative to demand; can be negative.",
        "use": "Possibly a battery / interconnection residual. Weak raw correlation with MCP is expected.",
        "watch": "Scale is tens of MW vs thousands of MW of demand — do not interpret coefficients on the same footing.",
    },
    "dpv_mw": {
        "role": "fundamental (behind-the-meter supply)",
        "meaning": "5-minute estimated distributed (rooftop) PV generation (MW).",
        "use": "Main daytime suppressor of net demand. Interacts with hour-of-day (duck curve).",
        "watch": "Almost a deterministic function of solar geometry + cloud. Prefer residual demand over stacking DPV and demand independently if they are collinear.",
    },
    "trading_interval_end": {
        "role": "join key",
        "meaning": "End of the parent 30-minute trading interval (six dispatch intervals share this stamp).",
        "use": "Not a regressor. Used only to attach RTP, STEM, and 30-min sent-out / DPV.",
        "watch": "30-min series are constant across the six 5-min rows — they do not have 5-min resolution.",
    },
    "rtp": {
        "role": "30-min energy price",
        "meaning": "Reference Trading Price ($/MWh) for the 30-minute trading interval.",
        "use": "Lagged RTP is a valid feature. Same-interval RTP is nearly a smoothed version of MCP and is leakage for 5-min MCP.",
        "watch": "Do not use contemporaneous RTP as a 'predictor' of MCP in the same trading interval.",
    },
    "stem_price": {
        "role": "day-ahead / STEM price",
        "meaning": "STEM clearing price ($/MWh). Forward market signal published before real-time dispatch.",
        "use": "Genuinely pre-dispatch information. One of the cleaner price-level features.",
        "watch": "STEM can be a poor guide on spike days; residuals vs STEM are themselves a feature.",
    },
    "stem_qty_mwh": {
        "role": "STEM volume",
        "meaning": "STEM clearing quantity (MWh).",
        "use": "Liquidity / cleared forward volume. Weak direct price signal.",
        "watch": "Collinear with stem_bid_mwh / stem_offer_mwh; keep one volume feature unless you engineer bid-offer imbalance.",
    },
    "stem_bid_mwh": {
        "role": "STEM volume",
        "meaning": "Total STEM bid quantity (MWh).",
        "use": "Demand-side of STEM. Bid–offer spread/imbalance is more interpretable than the raw level.",
        "watch": "Large levels, slow-moving. Standardise before linear models.",
    },
    "stem_offer_mwh": {
        "role": "STEM volume",
        "meaning": "Total STEM offer quantity (MWh).",
        "use": "Supply-side of STEM. Tight offer stacks can precede real-time scarcity.",
        "watch": "Same scale issue as stem_bid_mwh.",
    },
    "sent_out_mwh": {
        "role": "generation (30-min)",
        "meaning": "Total sent-out generation (MWh) over the 30-minute trading interval.",
        "use": "Drop in favour of sent_out_mw (exact linear map: MW = 2 × MWh).",
        "watch": "Exact collinearity with sent_out_mw. ~3% missing.",
    },
    "sent_out_mw": {
        "role": "generation (30-min)",
        "meaning": "Total sent-out generation expressed as average MW (2 × sent_out_mwh).",
        "use": "System-wide generation proxy at 30-min resolution. Collinear with demand / SCADA.",
        "watch": "~3% missing. Prefer 5-min SCADA or operational demand if both are available.",
    },
    "dpv_mw_30min": {
        "role": "DPV (30-min, legacy)",
        "meaning": "30-minute estimated DPV (MW). Coarser sibling of dpv_mw.",
        "use": "Prefer dpv_mw (5-min, near-complete). Keep 30-min only as a gap-fill.",
        "watch": "Highest missingness in the panel (~5%). Collinear with dpv_mw.",
    },
    "scada_mwh": {
        "role": "generation (SCADA)",
        "meaning": "Sum of facility SCADA average MWh over the 5-minute interval.",
        "use": "Drop in favour of scada_mw (exact linear map: MW = 12 × MWh).",
        "watch": "Exact collinearity with scada_mw.",
    },
    "scada_mw": {
        "role": "generation (SCADA)",
        "meaning": "Sum of facility SCADA converted to MW. Metered scheduled generation.",
        "use": "5-min generation proxy. Tracks operational demand closely.",
        "watch": "Does not include behind-the-meter DPV. Collinear with scheduled_energy_mw and demand.",
    },
    "scheduled_energy_mw": {
        "role": "dispatch schedule",
        "meaning": "Sum of market-schedule energy (MW) across facilities.",
        "use": "Pre-dispatch / schedule level. Useful if it is known before the interval; leakage if it is the realised schedule.",
        "watch": "Near-duplicate of scada_mw / operational demand. Check which vintage (forecast vs realised) the CSV is.",
    },
    "residual_demand_mw": {
        "role": "engineered fundamental",
        "meaning": "operational_demand_mw − dpv_mw. Net demand to be met by scheduled plant and storage.",
        "use": "Usually the single strongest fundamental for energy price. Prefer this over stacking demand and DPV raw.",
        "watch": "Exact linear combination of two other columns — do not include all three in a linear model.",
    },
    "mcp_lag1": {
        "role": "autoregressive lag",
        "meaning": "MCP one dispatch interval ago (5 minutes).",
        "use": "Strongest short-horizon predictor. Valid for 5-min-ahead; not available for day-ahead.",
        "watch": "Persistence will dominate a model that is evaluated on 5-min-ahead RMSE and hide whether fundamentals work.",
    },
    "mcp_lag6": {
        "role": "autoregressive lag",
        "meaning": "MCP six intervals ago (30 minutes, one trading interval).",
        "use": "Captures intra-hour persistence after lag-1 is included.",
        "watch": "Redundant with lag-1 in a linear model (high collinearity among lags).",
    },
    "mcp_lag12": {
        "role": "autoregressive lag",
        "meaning": "MCP twelve intervals ago (60 minutes).",
        "use": "Hour-ago price level. Useful when lag-1 is not allowed (longer horizon).",
        "watch": "Same collinearity note as other MCP lags.",
    },
    "hour": {
        "role": "calendar",
        "meaning": "Hour of interval_end (0–23), WEM clock time as stored.",
        "use": "Diurnal seasonality: morning ramp, midday DPV trough, evening peak.",
        "watch": "Cyclic — encode with sin/cos or as a categorical, not as a linear integer.",
    },
    "minute": {
        "role": "calendar",
        "meaning": "Minute of the dispatch interval (0, 5, …, 55).",
        "use": "Within-trading-interval position. Weak level effect; can matter for RTP-vs-MCP alignment.",
        "watch": "Only 12 unique values. Treat as categorical.",
    },
    "dow": {
        "role": "calendar",
        "meaning": "Day of week (0=Monday … 6=Sunday).",
        "use": "Weekend vs weekday load and price shapes.",
        "watch": "Collinear with is_weekend. Cyclic encoding or dummy encoding, not linear.",
    },
    "month": {
        "role": "calendar",
        "meaning": "Calendar month (1–12).",
        "use": "Seasonal demand, DPV, and gas/heat effects (WA summer peaks).",
        "watch": "Cyclic. Incomplete years at both ends of the sample will bias month means.",
    },
    "year": {
        "role": "calendar",
        "meaning": "Calendar year. Sample spans 2023–2026.",
        "use": "Regime / drift control. Do not extrapolate a linear year trend out of sample.",
        "watch": "2023 is October–December only; 2026 is partial. Year dummies are safer than a numeric trend.",
    },
    "is_weekend": {
        "role": "calendar",
        "meaning": "True iff dow ≥ 5 (Saturday/Sunday).",
        "use": "Simple weekday dummy. Prefer this plus public holidays (not yet in the panel) over raw dow if you need a parsimonious model.",
        "watch": "Deterministic function of dow.",
    },
}


def _require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Parquet support is missing. Activate the project venv:\n"
            '  cd "/Users/jg/Documents/MATH5001 Project"\n'
            "  source .venv/bin/activate\n"
            "  python scripts/explore_panel.py"
        ) from exc


def load_panel() -> pd.DataFrame:
    if not PANEL.exists():
        raise SystemExit(f"missing {PANEL}")
    _require_parquet_engine()
    df = pd.read_parquet(PANEL, engine="pyarrow")
    df["interval_end"] = pd.to_datetime(df["interval_end"])
    if "trading_interval_end" in df.columns:
        df["trading_interval_end"] = pd.to_datetime(df["trading_interval_end"])
    df = df.sort_values("interval_end").reset_index(drop=True)
    return df


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[c for c in DT_COLS if c in df.columns]).copy()
    if "is_weekend" in out.columns:
        out["is_weekend"] = out["is_weekend"].astype(int)
    return out.select_dtypes(include=[np.number])


def savefig(fig: plt.Figure, name: str) -> Path:
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_missingness(df: pd.DataFrame) -> None:
    miss = df.isna().mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * max(len(miss), 1) + 1.5)))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#4C78A8")
        ax.set_xlabel("Missing share (%)")
        ax.set_ylabel("")
        ax.set_title("Columns with missing values")
        for i, v in enumerate(miss.values):
            ax.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=9)
    savefig(fig, "01_missingness.png")


def plot_mcp_distribution(df: pd.DataFrame) -> dict[str, float]:
    y = df[TARGET].dropna()
    q = y.quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(y, bins=80, ax=axes[0, 0], color="#4C78A8")
    axes[0, 0].set_title("MCP — full range")
    axes[0, 0].set_xlabel("$/MWh")

    clipped = y.clip(q.loc[0.01], q.loc[0.99])
    sns.histplot(clipped, bins=80, ax=axes[0, 1], color="#F58518")
    axes[0, 1].set_title("MCP — 1st–99th percentile")
    axes[0, 1].set_xlabel("$/MWh")

    sns.boxplot(x=y, ax=axes[1, 0], color="#4C78A8")
    axes[1, 0].set_title("MCP boxplot (full)")
    axes[1, 0].set_xlabel("$/MWh")

    sns.ecdfplot(y, ax=axes[1, 1], color="#54A24B")
    axes[1, 1].set_title("MCP empirical CDF")
    axes[1, 1].set_xlabel("$/MWh")
    fig.suptitle("Target: energy market clearing price (MCP)", y=1.02)
    fig.tight_layout()
    savefig(fig, "02_mcp_distribution.png")
    return {f"q{int(k*1000)/10:.1f}": float(v) for k, v in q.items()} | {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "skew": float(y.skew()),
        "kurtosis": float(y.kurtosis()),
        "min": float(y.min()),
        "max": float(y.max()),
        "neg_share": float((y < 0).mean()),
        "zero_share": float((y == 0).mean()),
    }


def plot_mcp_timeseries(df: pd.DataFrame) -> None:
    daily = df.set_index("interval_end")[TARGET].resample("D").agg(["median", "mean", "min", "max"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(daily.index, daily["median"], lw=1.1, label="daily median", color="#4C78A8")
    axes[0].plot(daily.index, daily["mean"], lw=0.9, alpha=0.7, label="daily mean", color="#F58518")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("$/MWh")
    axes[0].set_title("Daily MCP level")
    axes[1].fill_between(daily.index, daily["min"], daily["max"], color="#4C78A8", alpha=0.35)
    axes[1].set_ylabel("$/MWh")
    axes[1].set_title("Daily MCP range (min–max)")
    fig.tight_layout()
    savefig(fig, "03_mcp_timeseries.png")


def plot_correlation_heatmaps(num: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson = num.corr(method="pearson")
    spearman = num.corr(method="spearman")
    for name, mat, file in (
        ("Pearson", pearson, "04_corr_pearson.png"),
        ("Spearman", spearman, "05_corr_spearman.png"),
    ):
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            mat,
            ax=ax,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.2,
            cbar_kws={"shrink": 0.7, "label": f"{name} r"},
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_title(f"{name} correlation — all numeric columns")
        plt.xticks(rotation=55, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        fig.tight_layout()
        savefig(fig, file)

    # Clustered Spearman is better for grouping redundant blocks.
    cg = sns.clustermap(
        spearman.fillna(0),
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(16, 16),
        dendrogram_ratio=0.12,
        linewidths=0.2,
    )
    cg.fig.suptitle("Spearman correlation (clustered)", y=1.01)
    cg.savefig(FIG / "06_corr_spearman_clustered.png", dpi=140, bbox_inches="tight")
    plt.close(cg.fig)
    return pearson, spearman


def plot_corr_with_target(pearson: pd.DataFrame, spearman: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in pearson.columns:
        if col == TARGET:
            continue
        rows.append(
            {
                "column": col,
                "pearson": pearson.loc[TARGET, col],
                "spearman": spearman.loc[TARGET, col],
            }
        )
    tab = pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
    long = tab.melt(id_vars="column", value_vars=["pearson", "spearman"], var_name="method", value_name="r")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(tab) + 1)))
    sns.barplot(data=long, y="column", x="r", hue="method", ax=ax, palette=["#4C78A8", "#F58518"])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Correlation with MCP")
    ax.set_ylabel("")
    ax.set_title("Each feature vs MCP — Pearson vs Spearman")
    fig.tight_layout()
    savefig(fig, "07_corr_with_mcp.png")
    return tab


def plot_univariate_grid(num: pd.DataFrame) -> None:
    cols = list(num.columns)
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.6 * nrows))
    axes = np.ravel(axes)
    for i, col in enumerate(cols):
        ax = axes[i]
        s = num[col].dropna()
        nunique = s.nunique()
        if nunique <= 24:
            order = np.sort(s.unique())
            sns.countplot(x=s, order=order, ax=ax, color="#4C78A8")
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        else:
            lo, hi = s.quantile([0.005, 0.995])
            sns.histplot(s.clip(lo, hi), bins=40, ax=ax, color="#4C78A8")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle("Univariate distributions (continuous clipped to 0.5–99.5 pct)", y=1.01)
    fig.tight_layout()
    savefig(fig, "08_univariate_grid.png")


def plot_calendar(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.boxplot(data=df, x="hour", y=TARGET, ax=axes[0, 0], showfliers=False, color="#4C78A8")
    axes[0, 0].set_title("MCP by hour (fliers hidden)")
    axes[0, 0].set_ylabel("$/MWh")

    sns.boxplot(data=df, x="dow", y=TARGET, ax=axes[0, 1], showfliers=False, color="#F58518")
    axes[0, 1].set_title("MCP by weekday (0=Mon)")
    axes[0, 1].set_ylabel("$/MWh")

    sns.boxplot(data=df, x="month", y=TARGET, ax=axes[1, 0], showfliers=False, color="#54A24B")
    axes[1, 0].set_title("MCP by month (fliers hidden)")
    axes[1, 0].set_ylabel("$/MWh")

    sns.boxplot(data=df, x="is_weekend", y=TARGET, ax=axes[1, 1], showfliers=False, color="#E45756")
    axes[1, 1].set_title("MCP weekday vs weekend")
    axes[1, 1].set_ylabel("$/MWh")
    fig.tight_layout()
    savefig(fig, "09_mcp_calendar_boxplots.png")

    pivot = df.pivot_table(index="hour", columns="dow", values=TARGET, aggfunc="median")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot, ax=ax, cmap="mako", annot=False, cbar_kws={"label": "median MCP ($/MWh)"})
    ax.set_xlabel("dow (0=Mon)")
    ax.set_title("Median MCP — hour × weekday")
    fig.tight_layout()
    savefig(fig, "10_mcp_hour_dow_heatmap.png")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    hourly = df.groupby("hour")[[TARGET, "dpv_mw", "operational_demand_mw", "residual_demand_mw"]].median()
    ax2 = ax.twinx()
    ax.plot(hourly.index, hourly[TARGET], color="#E45756", lw=2, label="median MCP")
    ax2.plot(hourly.index, hourly["operational_demand_mw"], color="#4C78A8", lw=1.4, label="demand")
    ax2.plot(hourly.index, hourly["dpv_mw"], color="#F58518", lw=1.4, label="DPV")
    ax2.plot(hourly.index, hourly["residual_demand_mw"], color="#54A24B", lw=1.4, label="residual demand")
    ax.set_xlabel("hour")
    ax.set_ylabel("MCP ($/MWh)")
    ax2.set_ylabel("MW")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title("Typical diurnal shape: price vs demand / DPV")
    fig.tight_layout()
    savefig(fig, "11_diurnal_shape.png")


def plot_scatters(df: pd.DataFrame, sample: pd.DataFrame) -> None:
    pairs = [
        ("residual_demand_mw", "Net / residual demand vs MCP"),
        ("operational_demand_mw", "Operational demand vs MCP"),
        ("dpv_mw", "Distributed PV vs MCP"),
        ("stem_price", "STEM price vs MCP"),
        ("rtp", "RTP vs MCP (same 30-min interval — leakage risk)"),
        ("scada_mw", "SCADA generation vs MCP"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    y = sample[TARGET]
    for ax, (xcol, title) in zip(axes, pairs):
        if xcol not in sample.columns:
            ax.set_axis_off()
            continue
        sns.scatterplot(x=sample[xcol], y=y, ax=ax, s=8, alpha=0.15, edgecolor=None, color="#4C78A8")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("MCP")
    fig.suptitle(f"Fundamentals vs MCP (random {len(sample):,} rows)", y=1.01)
    fig.tight_layout()
    savefig(fig, "12_scatter_fundamentals.png")

    key = [
        c
        for c in [
            TARGET,
            "residual_demand_mw",
            "operational_demand_mw",
            "dpv_mw",
            "stem_price",
            "rtp",
            "scada_mw",
            "mcp_lag1",
        ]
        if c in sample.columns
    ]
    pair_n = min(4000, len(sample))
    g = sns.pairplot(
        sample[key].sample(pair_n, random_state=7),
        corner=True,
        diag_kind="kde",
        plot_kws={"s": 8, "alpha": 0.2, "edgecolor": None},
    )
    g.fig.suptitle(f"Key-feature pairplot (n={pair_n:,})", y=1.02)
    g.savefig(FIG / "13_pairplot_key.png", dpi=120, bbox_inches="tight")
    plt.close(g.fig)

    collinear = [
        c
        for c in [
            "operational_demand_mw",
            "unscheduled_demand_mw",
            "scada_mw",
            "scheduled_energy_mw",
            "sent_out_mw",
            "residual_demand_mw",
        ]
        if c in df.columns
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        df[collinear].corr(method="spearman"),
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
    )
    ax.set_title("Demand / generation block — Spearman (collinearity)")
    fig.tight_layout()
    savefig(fig, "14_demand_collinearity.png")

    fcas = [c for c in ["cr_raise", "cr_lower", "reg_raise", "reg_lower", "rocof", TARGET] if c in df.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        df[fcas].corr(method="spearman"),
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
    )
    ax.set_title("FCAS prices vs MCP — Spearman")
    fig.tight_layout()
    savefig(fig, "15_fcas_correlation.png")


def linear_identities(df: pd.DataFrame) -> list[str]:
    notes = []
    if {"sent_out_mw", "sent_out_mwh"} <= set(df.columns):
        gap = (df["sent_out_mw"] - 2.0 * df["sent_out_mwh"]).abs().max(skipna=True)
        notes.append(f"sent_out_mw − 2×sent_out_mwh  max |error| = {gap:.3g}")
    if {"scada_mw", "scada_mwh"} <= set(df.columns):
        gap = (df["scada_mw"] - 12.0 * df["scada_mwh"]).abs().max(skipna=True)
        notes.append(f"scada_mw − 12×scada_mwh  max |error| = {gap:.3g}")
    if {"residual_demand_mw", "operational_demand_mw", "dpv_mw"} <= set(df.columns):
        gap = (df["residual_demand_mw"] - (df["operational_demand_mw"] - df["dpv_mw"])).abs().max(skipna=True)
        notes.append(f"residual_demand_mw − (demand − DPV)  max |error| = {gap:.3g}")
    if {"is_weekend", "dow"} <= set(df.columns):
        ok = bool((df["is_weekend"].astype(int) == (df["dow"] >= 5).astype(int)).all())
        notes.append(f"is_weekend == (dow ≥ 5)  identity holds: {ok}")
    return notes


def mutual_info(num: pd.DataFrame, n: int = 40000) -> pd.Series:
    work = num.dropna(subset=[TARGET])
    if len(work) > n:
        work = work.sample(n, random_state=7)
    x = work.drop(columns=[TARGET])
    x = x.fillna(x.median(numeric_only=True))
    y = work[TARGET].to_numpy()
    mi = mutual_info_regression(x, y, random_state=7, n_neighbors=5)
    return pd.Series(mi, index=x.columns).sort_values(ascending=False)


def plot_mutual_info(mi: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(mi) + 1)))
    sns.barplot(x=mi.values, y=mi.index, ax=ax, color="#72B7B2")
    ax.set_xlabel("Mutual information with MCP (nats, subsample)")
    ax.set_ylabel("")
    ax.set_title("Non-linear association with MCP")
    fig.tight_layout()
    savefig(fig, "16_mutual_info.png")


def column_profile(df: pd.DataFrame, num: pd.DataFrame, col: str) -> dict:
    s = df[col]
    profile: dict = {
        "column": col,
        "dtype": str(s.dtype),
        "n": int(len(s)),
        "n_missing": int(s.isna().sum()),
        "missing_pct": float(100 * s.isna().mean()),
        "n_unique": int(s.nunique(dropna=True)),
    }
    if col in num.columns:
        x = num[col]
        profile.update(
            {
                "mean": float(x.mean()) if x.notna().any() else np.nan,
                "std": float(x.std()) if x.notna().any() else np.nan,
                "min": float(x.min()) if x.notna().any() else np.nan,
                "p01": float(x.quantile(0.01)) if x.notna().any() else np.nan,
                "median": float(x.median()) if x.notna().any() else np.nan,
                "p99": float(x.quantile(0.99)) if x.notna().any() else np.nan,
                "max": float(x.max()) if x.notna().any() else np.nan,
                "skew": float(x.skew()) if x.notna().any() else np.nan,
                "zero_pct": float(100 * (x == 0).mean()),
                "negative_pct": float(100 * (x < 0).mean()),
            }
        )
    return profile


def fmt(v, digits=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.{digits}g}"
    return str(v)


def write_report(
    df: pd.DataFrame,
    num: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    vs_mcp: pd.DataFrame,
    mi: pd.Series,
    identities: list[str],
    mcp_stats: dict,
) -> Path:
    profiles = [column_profile(df, num, c) for c in df.columns]
    prof = {p["column"]: p for p in profiles}
    pd.DataFrame(profiles).to_csv(OUT / "column_profiles.csv", index=False)
    pearson.to_csv(OUT / "corr_pearson.csv")
    spearman.to_csv(OUT / "corr_spearman.csv")
    vs_mcp.to_csv(OUT / "corr_with_mcp.csv", index=False)
    mi.rename("mutual_info").to_csv(OUT / "mutual_info_with_mcp.csv")

    lines: list[str] = []
    a = lines.append
    a("# WEM 5-minute panel — correlation and feature interpretation")
    a("")
    a("Generated by `scripts/explore_panel.py` from the **full** processed parquet.")
    a("")
    a("## Panel overview")
    a("")
    a(f"- File: `{PANEL.relative_to(ROOT)}`")
    a(f"- Rows: **{len(df):,}**")
    a(f"- Columns: **{len(df.columns)}**")
    a(f"- Range: **{df['interval_end'].min()} → {df['interval_end'].max()}**")
    expected = pd.date_range(df["interval_end"].min(), df["interval_end"].max(), freq="5min")
    missing_slots = expected.difference(df["interval_end"])
    a(f"- Missing 5-minute slots in the calendar span: **{len(missing_slots):,}**")
    a("")
    a("## Target (MCP) snapshot")
    a("")
    a("| stat | value |")
    a("|---|---|")
    for k, label in (
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("skew", "skewness"),
        ("kurtosis", "excess kurtosis"),
        ("neg_share", "share < 0"),
    ):
        a(f"| {label} | {fmt(mcp_stats[k])} |")
    a("")
    a(
        "MCP is heavy-tailed (electricity prices spike). **Spearman** and **mutual information** "
        "are more trustworthy than Pearson for ranking features against MCP. Pearson is still "
        "useful for spotting *linear collinearity* among demand/generation columns."
    )
    a("")
    a("## Exact linear identities (do not include both sides in a linear model)")
    a("")
    for note in identities:
        a(f"- `{note}`")
    a("")
    a("Recommended drops for linear / GLM-style models:")
    a("")
    a("- `sent_out_mwh` (keep `sent_out_mw`)")
    a("- `scada_mwh` (keep `scada_mw`)")
    a("- `unscheduled_demand_mw` (keep `operational_demand_mw`)")
    a("- either `residual_demand_mw` **or** the pair (`operational_demand_mw`, `dpv_mw`), not all three")
    a("- `is_weekend` **or** `dow`, not both as numeric")
    a("- contemporaneous `rtp` if the task is to forecast MCP (leakage)")
    a("")
    a("## Association with MCP — ranked")
    a("")
    a("| rank | column | Spearman | Pearson | MI (subsample) |")
    a("|---:|---|---:|---:|---:|")
    mi_map = mi.to_dict()
    for i, row in enumerate(vs_mcp.itertuples(index=False), start=1):
        a(
            f"| {i} | `{row.column}` | {row.spearman:.3f} | {row.pearson:.3f} | "
            f"{mi_map.get(row.column, float('nan')):.3f} |"
        )
    a("")
    a("## Figures")
    a("")
    for name, caption in (
        ("01_missingness.png", "Missingness"),
        ("02_mcp_distribution.png", "MCP distribution"),
        ("03_mcp_timeseries.png", "Daily MCP time series"),
        ("04_corr_pearson.png", "Pearson heatmap"),
        ("05_corr_spearman.png", "Spearman heatmap"),
        ("06_corr_spearman_clustered.png", "Clustered Spearman heatmap"),
        ("07_corr_with_mcp.png", "Each column vs MCP"),
        ("08_univariate_grid.png", "Univariate distributions"),
        ("09_mcp_calendar_boxplots.png", "MCP by calendar"),
        ("10_mcp_hour_dow_heatmap.png", "Hour × weekday median MCP"),
        ("11_diurnal_shape.png", "Diurnal price vs demand/DPV"),
        ("12_scatter_fundamentals.png", "Scatter vs MCP"),
        ("13_pairplot_key.png", "Key-feature pairplot"),
        ("14_demand_collinearity.png", "Demand/generation collinearity"),
        ("15_fcas_correlation.png", "FCAS vs MCP"),
        ("16_mutual_info.png", "Mutual information with MCP"),
    ):
        a(f"### {caption}")
        a("")
        a(f"![{caption}](figures/{name})")
        a("")

    a("## Per-column interpretation")
    a("")
    a(
        "Each section below combines the construction notes from `scripts/build_panel.py` "
        "with full-sample descriptive statistics and the strongest Spearman neighbours."
    )
    a("")

    for col in df.columns:
        doc = COLUMN_DOCS.get(col, {})
        p = prof[col]
        a(f"### `{col}`")
        a("")
        a(f"- **Role:** {doc.get('role', 'see stats')}")
        a(f"- **Meaning:** {doc.get('meaning', '—')}")
        a(
            f"- **dtype:** `{p['dtype']}` · unique={fmt(p['n_unique'])} · "
            f"missing={p['missing_pct']:.2f}% ({fmt(p['n_missing'])} rows)"
        )
        if col in num.columns:
            a(
                f"- **Distribution:** min={fmt(p['min'])}, p01={fmt(p['p01'])}, "
                f"median={fmt(p['median'])}, mean={fmt(p['mean'])}, p99={fmt(p['p99'])}, "
                f"max={fmt(p['max'])}, std={fmt(p['std'])}, skew={fmt(p['skew'])}"
            )
            a(f"- **Signs:** {p['zero_pct']:.2f}% zeros, {p['negative_pct']:.2f}% negative")
            if col != TARGET and col in spearman.columns:
                r_s = spearman.loc[TARGET, col]
                r_p = pearson.loc[TARGET, col]
                a(f"- **vs MCP:** Spearman = {r_s:.3f}, Pearson = {r_p:.3f}, MI = {mi_map.get(col, float('nan')):.3f}")
            neigh = (
                spearman[col]
                .drop(labels=[col], errors="ignore")
                .abs()
                .sort_values(ascending=False)
                .head(6)
            )
            bits = [f"`{n}` ({spearman.loc[col, n]:+.2f})" for n in neigh.index]
            a(f"- **Closest Spearman neighbours:** {', '.join(bits)}")
        a(f"- **How to use:** {doc.get('use', '—')}")
        a(f"- **Watch-outs:** {doc.get('watch', '—')}")
        a("")

    a("## Modelling takeaways")
    a("")
    a("1. **Horizon first.** `mcp_lag1` will crush a 5-minute-ahead point forecast and is illegal for day-ahead. Split feature sets by horizon.")
    a("2. **Do not leak RTP.** Same-interval `rtp` is a 30-minute energy price built from the same market as MCP.")
    a("3. **Collapse the demand block.** Keep `residual_demand_mw` (or demand + DPV) plus at most one of SCADA / scheduled / sent-out.")
    a("4. **STEM is the clean forward price.** `stem_price` is the least-leaky price-level covariate in the panel.")
    a("5. **Calendar is nonlinear.** Encode `hour`, `dow`, `month` as cyclic or categorical; `year` is a regime dummy, not a trend to extrapolate.")
    a("6. **FCAS is a separate market.** Use lagged FCAS, or model it separately; same-interval FCAS is not an energy fundamental.")
    a("7. **Score on ranks and tails.** High kurtosis in MCP means pinball / interval coverage / CRPS should lead; RMSE is spike-dominated.")
    a("8. **Fill or drop 30-min generation gaps.** `dpv_mw_30min` (~5% missing) and `sent_out_*` (~3% missing) are the only material holes.")
    a("")

    path = OUT / "feature_interpretation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    FIG.mkdir(parents=True, exist_ok=True)

    print(f"loading {PANEL}")
    df = load_panel()
    print(f"rows={len(df):,}  cols={len(df.columns)}  {df['interval_end'].min()} → {df['interval_end'].max()}")

    num = numeric_frame(df)
    sample = df.sample(n=min(20_000, len(df)), random_state=7)

    print("missingness + MCP distribution/time series")
    plot_missingness(df)
    mcp_stats = plot_mcp_distribution(df)
    plot_mcp_timeseries(df)

    print("correlation heatmaps")
    pearson, spearman = plot_correlation_heatmaps(num)
    vs_mcp = plot_corr_with_target(pearson, spearman)

    print("univariate grid + calendar")
    plot_univariate_grid(num)
    plot_calendar(df)

    print("scatters / pairplot / collinearity")
    plot_scatters(df, sample)

    print("mutual information")
    mi = mutual_info(num)
    plot_mutual_info(mi)

    identities = linear_identities(df)
    print("identities:")
    for note in identities:
        print(" ", note)

    path = write_report(df, num, pearson, spearman, vs_mcp, mi, identities, mcp_stats)
    print(f"wrote {path}")
    print(f"figures in {FIG}")
    print("top Spearman |r| with MCP:")
    print(vs_mcp.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
