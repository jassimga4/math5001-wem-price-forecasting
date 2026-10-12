"""Conformal predictive systems (CPS) for the spike forecaster, and the final comparison.

Every CPS here is built on the calibration window only (2025-10-01 to
2026-03-31) and frozen for test. Variants:

* ``cps_split``: split CPS (Vovk et al., 2019) on the regime-switch point (the
  spike forecaster's point). The predictive CDF at an origin is
  Q(y) = (#{r_i < y - yhat} + tau (#{r_i = y - yhat} + 1)) / (n + 1) over the n
  calibration residuals r_i = y_i - yhat_i. The project's sliding 7-day residual
  window (W = 2016) around the same point is the regime-switch row.
* ``cps_mondrian``: Mondrian split CPS on the same point. Calibration residuals
  are grouped into spike-risk categories known at the origin: the last price
  already above the spike threshold, already below the floor, or inside the
  band and binned by the forecaster's P(spike up). A test origin uses the
  residuals of its own category, so high-risk origins get the heavy right tail
  that spikes need, and quiet origins a narrow one.
* ``cps_pit`` and ``cps_pit_mondrian``: CPS on the forecaster's own predictive
  distribution F, with the PIT value F(y) as conformity score (conformal
  recalibration). The recalibrated CDF is G(F(y)), with G the calibration PITs'
  empirical CDF (global, or per risk category).

Isotonic regression recalibrates P(spike up) on calibration.

Calibration-period scores are cross-fitted: the calibration window is split in
two halves by time, and rows in each half are scored with a CPS built on the
other half. Test rows use the CPS built on all calibration rows.

Run: ``python scripts/spike_cps.py`` (writes reports/forecast/final_*.csv).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import spike_onset as so  # noqa: E402
from scripts.conformal import lower_rank, sliding_residual_quantiles, upper_rank  # noqa: E402
from scripts.metrics import crps_from_residual_samples  # noqa: E402

OUT = ROOT / "reports" / "forecast"
BASE_CACHE = ROOT / "data" / "processed" / "cps_base_rows.parquet"
ALPHA = 0.10
WINDOW = 2016
RISK_EDGES = (0.01, 0.05, 0.2)  # P(spike up) bins for eligible origins, fixed before looking at any score
MIN_CATEGORY = 300  # calibration residuals per category; smaller categories merge into the next lower risk bin
TAU = 0.5  # CPS tie-break / randomisation constant (mid-point)


# --------------------------------------------------------------------------- core CPS


class SplitCPS:
    """Split CPS on additive residuals: predictive distribution yhat + {r_1..r_n}."""

    def __init__(self, residuals):
        r = np.asarray(residuals, dtype=float)
        r = np.sort(r[np.isfinite(r)])
        if len(r) == 0:
            raise ValueError("no calibration residuals")
        self.r, self.n = r, len(r)
        self.cum = np.concatenate([[0.0], np.cumsum(r)])
        k = np.arange(1, self.n + 1)
        self.pair = float(np.sum((2 * k - self.n - 1) * r) / self.n ** 2)

    def crps(self, y, yhat) -> np.ndarray:
        """Exact CRPS of the empirical distribution yhat + r (same as metrics.crps_from_residual_samples)."""
        z = np.asarray(y, float) - np.asarray(yhat, float)
        k = np.searchsorted(self.r, z, side="right")
        below = k * z - self.cum[k]
        above = (self.cum[-1] - self.cum[k]) - (self.n - k) * z
        return (below + above) / self.n - self.pair

    def cdf(self, y, yhat, tau: float = TAU) -> np.ndarray:
        z = np.asarray(y, float) - np.asarray(yhat, float)
        lo = np.searchsorted(self.r, z, side="left")
        hi = np.searchsorted(self.r, z, side="right")
        return (lo + tau * (hi - lo + 1)) / (self.n + 1)

    def interval(self, yhat, alpha: float = ALPHA):
        yhat = np.asarray(yhat, float)
        return yhat + self.r[lower_rank(self.n, alpha)], yhat + self.r[upper_rank(self.n, alpha)]

    def quantile(self, yhat, level: float) -> np.ndarray:
        idx = min(max(int(np.ceil(level * (self.n + 1))), 1), self.n) - 1
        return np.asarray(yhat, float) + self.r[idx]


class MondrianCPS:
    """One SplitCPS per category; categories without calibration data use the pooled CPS."""

    def __init__(self, residuals, categories):
        residuals, categories = np.asarray(residuals, float), np.asarray(categories)
        self.pooled = SplitCPS(residuals)
        self.parts = {c: SplitCPS(residuals[categories == c]) for c in np.unique(categories) if c >= 0}

    def _apply(self, categories, fn):
        categories = np.asarray(categories)
        out = None
        for c in np.unique(categories):
            m = categories == c
            part = fn(self.parts.get(c, self.pooled), m)
            if out is None:
                out = tuple(np.empty(len(categories)) for _ in part) if isinstance(part, tuple) else np.empty(len(categories))
            if isinstance(part, tuple):
                for o, p in zip(out, part):
                    o[m] = p
            else:
                out[m] = part
        return out

    def crps(self, y, yhat, categories):
        y, yhat = np.asarray(y, float), np.asarray(yhat, float)
        return self._apply(categories, lambda cps, m: cps.crps(y[m], yhat[m]))

    def cdf(self, y, yhat, categories, tau: float = TAU):
        y, yhat = np.asarray(y, float), np.asarray(yhat, float)
        return self._apply(categories, lambda cps, m: cps.cdf(y[m], yhat[m], tau))

    def interval(self, yhat, categories, alpha: float = ALPHA):
        yhat = np.asarray(yhat, float)
        return self._apply(categories, lambda cps, m: cps.interval(yhat[m], alpha))


class PITCalibrator:
    """Conformal recalibration of a predictive CDF F through its calibration PIT values."""

    def __init__(self, pits):
        u = np.asarray(pits, float)
        self.u = np.sort(u[np.isfinite(u)])
        self.m = len(self.u)

    def G(self, c) -> np.ndarray:
        """Empirical CDF of the calibration PITs (step function, G(0) = 0, G(1) = 1)."""
        return np.searchsorted(self.u, np.asarray(c, float), side="right") / self.m

    def cdf(self, f_y, tau: float = TAU) -> np.ndarray:
        lo = np.searchsorted(self.u, f_y, side="left")
        hi = np.searchsorted(self.u, f_y, side="right")
        return (lo + tau * (hi - lo + 1)) / (self.m + 1)

    def levels(self, alpha: float = ALPHA):
        return self.u[lower_rank(self.m, alpha)], self.u[upper_rank(self.m, alpha)]


def risk_category(eligible, p_up, last, spike: float, floor: float, edges=RISK_EDGES) -> np.ndarray:
    """Spike-risk category known at the origin: 0 = last price at/above spike, 1 = at/below floor,
    2.. = eligible origins binned by P(spike up)."""
    eligible, last = np.asarray(eligible, bool), np.asarray(last, float)
    cat = np.where(last >= spike, 0, 1)
    p = np.nan_to_num(np.asarray(p_up, float), nan=0.0)
    return np.where(eligible, 2 + np.digitize(p, edges), cat).astype(int)


def merge_small(cal_categories, min_size: int = MIN_CATEGORY) -> dict:
    """Category map fixed from calibration counts only.

    An eligible P(up) bin with fewer than ``min_size`` calibration rows joins the next lower-risk bin; a tail
    category that small maps to -1, which uses the pooled residuals.
    """
    counts = pd.Series(np.asarray(cal_categories)).value_counts()
    mapping = {}
    for c in range(0, 3 + len(RISK_EDGES)):
        if c < 2:
            mapping[c] = c if counts.get(c, 0) >= min_size else -1
            continue
        target, total = c, counts.get(c, 0)
        while total < min_size and target > 2:
            target -= 1
            total = sum(counts.get(k, 0) for k in range(target, c + 1))
        mapping[c] = target
    # a bin merged downward takes every higher bin that merged into it
    return mapping


def mixture_pit(x_sorted, w_sorted, y, tau: float = TAU) -> np.ndarray:
    """F(y-) + tau P(y) for a weighted empirical distribution (rows sorted ascending)."""
    w = w_sorted / w_sorted.sum(axis=1, keepdims=True)
    y = np.asarray(y, float)[:, None]
    return (w * (x_sorted < y)).sum(axis=1) + tau * (w * (x_sorted == y)).sum(axis=1)


def quantile_rows(x_sorted, cdf_sorted, levels) -> np.ndarray:
    """Left-continuous inverse CDF with one level per row."""
    idx = np.minimum((cdf_sorted < np.asarray(levels, float)[:, None] - 1e-12).sum(axis=1), x_sorted.shape[1] - 1)
    return x_sorted[np.arange(len(x_sorted)), idx]


def recalibrated_weights(cdf_sorted, calibrator: PITCalibrator) -> np.ndarray:
    g = calibrator.G(cdf_sorted)
    return np.diff(np.concatenate([np.zeros((len(g), 1)), g], axis=1), axis=1)


# --------------------------------------------------------------------------- scoring


def interval_score(y, lo, hi, alpha: float = ALPHA) -> np.ndarray:
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def score_block(name, rows, crps, lo, hi, point=None) -> list[dict]:
    """Metrics by regime: all, body, tail and onset (first interval of an event); fresh onsets for CRPS."""
    y = rows["y"].to_numpy(float)
    groups = {"all": np.ones(len(rows), bool), "body": ~rows["tail"].to_numpy(bool), "tail": rows["tail"].to_numpy(bool),
              "onset": rows["onset"].to_numpy(bool), "fresh_onset": rows["fresh_onset"].to_numpy(bool)}
    out = []
    covered = (y >= lo) & (y <= hi)
    iscore = interval_score(y, lo, hi)
    for g, m in groups.items():
        out.append({"model": name, "regime": g, "n": int(m.sum()), "crps": float(np.mean(crps[m])),
                    "coverage_90": float(covered[m].mean()), "width_90": float(np.mean((hi - lo)[m])),
                    "interval_score_90": float(iscore[m].mean()),
                    "mae": float(np.mean(np.abs(y - point)[m])) if point is not None else np.nan})
    return out


# --------------------------------------------------------------------------- data


def build_base(cache: Path = BASE_CACHE) -> tuple[pd.DataFrame, dict]:
    """Per-row inputs for calibration and test: forecaster outputs, PITs, sliding baselines."""
    from scripts import spike_forecaster as sf

    model = sf.SpikeForecaster.load()
    cfg = model.config
    frame = sf.build_frame(sf.read_panel(), cfg)
    model.set_history(frame)
    rows = frame.loc[frame["split"].isin(["calibration", "test"]) & frame["y"].notna()].copy()
    prepared, history, positions = model._prepare(rows)
    rows["p_up"], rows["p_down"] = prepared["p_up"].to_numpy(), prepared["p_down"].to_numpy()
    rows["fc_crps"] = model.crps(rows)
    n = len(rows)
    pit, q05, q95, q50 = (np.empty(n) for _ in range(4))
    for start in range(0, n, 2048):
        idx = np.arange(start, min(start + 2048, n))
        samples, weights = model._mixture(prepared, history, positions, idx)
        order = np.argsort(samples, axis=1)
        x, w = np.take_along_axis(samples, order, axis=1), np.take_along_axis(weights, order, axis=1)
        pit[idx] = mixture_pit(x, w, rows["y"].to_numpy(float)[idx])
        c = np.cumsum(w, axis=1) / w.sum(axis=1, keepdims=True)
        for arr, lev in ((q05, 0.05), (q50, 0.5), (q95, 0.95)):
            arr[idx] = quantile_rows(x, c, np.full(len(idx), lev))
    rows["fc_pit"], rows["fc_q05"], rows["fc_q50"], rows["fc_q95"] = pit, q05, q50, q95
    # sliding W = 2016 absolute-residual CPS around persistence, LightGBM and the regime switch (frozen conformal)
    realised = frame.loc[frame["y"].notna()]
    pos = np.searchsorted(realised.index.to_numpy(), rows.index.to_numpy(), side="left")
    for col in ("persistence", "lightgbm", "regime"):
        hist = (realised["y"] - realised[col]).to_numpy(float)
        res = so.residual_samples(hist, pos)
        z = rows["y"].to_numpy(float) - rows[col].to_numpy(float)
        rows[f"{col}_crps"] = crps_from_residual_samples(res, rows["y"].to_numpy(float), rows[col].to_numpy(float))
        lo, hi = sliding_residual_quantiles(hist, pos, WINDOW, ALPHA)
        rows[f"{col}_lo"], rows[f"{col}_hi"] = rows[col].to_numpy(float) + lo, rows[col].to_numpy(float) + hi
        less = (res < z[:, None]).sum(axis=1)
        ties = (res == z[:, None]).sum(axis=1)
        rows[f"{col}_pit"] = (less + TAU * (ties + 1)) / (WINDOW + 1)
    keep = ["split", "y", "last", "persistence", "lightgbm", "regime", "eligible", "tail", "onset", "fresh_onset",
            "rest_of_event", "up", "down", "p_up", "p_down", "fc_crps", "fc_pit", "fc_q05", "fc_q50", "fc_q95"]
    keep += [c for c in rows.columns if c.endswith(("_crps", "_lo", "_hi", "_pit")) and c not in keep]
    base = rows[keep].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(cache)
    return base, cfg


def scored_mask(base: pd.DataFrame) -> np.ndarray:
    """Calibration rows after the first residual window (as in spike_onset), and every test row."""
    is_cal = base["split"].eq("calibration").to_numpy()
    cal_pos = np.cumsum(is_cal) - 1
    return (~is_cal) | (cal_pos >= WINDOW)


def fold_labels(base: pd.DataFrame) -> np.ndarray:
    """'A' / 'B' halves of calibration by time; test rows are 'T'."""
    is_cal = base["split"].eq("calibration").to_numpy()
    cal_index = base.index[is_cal]
    cut = cal_index[len(cal_index) // 2]
    return np.where(~is_cal, "T", np.where(base.index < cut, "A", "B"))


# --------------------------------------------------------------------------- main


SLIDING_WINDOWS = {2: WINDOW}  # low-risk bin: the project's frozen 7-day window; other categories use MIN_CATEGORY
MIN_SLIDING = 30  # fewer earlier residuals in a category than this: use the pooled sliding window


def sliding_mondrian(base: pd.DataFrame, cat: np.ndarray):
    """Mondrian CPS with a sliding window per category.

    For an origin in category c the residual set is the last K_c realised regime-switch residuals of category c
    strictly before the target (K = 2016 for the low-risk bin, MIN_CATEGORY otherwise), from calibration and
    test rows already realised by the origin, as the project's frozen sliding conformal does. Categories with
    fewer than MIN_SLIDING earlier residuals fall back to the pooled last-2016 window.
    """
    y = base["y"].to_numpy(float)
    point = base["regime"].to_numpy(float)
    resid = y - point
    n = len(base)
    out = {k: np.full(n, np.nan) for k in ("crps", "lo", "hi", "pit")}
    times = np.arange(n)  # base rows are time-ordered; position i is strictly before i + 1
    pooled_pos = times
    for c in np.unique(cat):
        rows = np.flatnonzero(cat == c)
        members = rows if c >= 0 else np.array([], dtype=int)
        k_c = SLIDING_WINDOWS.get(c, MIN_CATEGORY)
        before = np.searchsorted(members, rows, side="left")  # members strictly before each row
        for i, row in enumerate(rows):
            have = before[i]
            if have >= MIN_SLIDING:
                sample = resid[members[max(0, have - k_c):have]]
            elif row >= WINDOW:
                sample = resid[pooled_pos[row - WINDOW:row]]
            else:
                continue
            cps_ = SplitCPS(sample)
            out["crps"][row] = cps_.crps(y[row:row + 1], point[row:row + 1])[0]
            out["pit"][row] = cps_.cdf(y[row:row + 1], point[row:row + 1])[0]
            lo, hi = cps_.interval(point[row:row + 1])
            out["lo"][row], out["hi"][row] = lo[0], hi[0]
    return out


def residual_cps(base: pd.DataFrame, cat: np.ndarray, folds: np.ndarray):
    """Split and Mondrian CPS on the regime-switch residuals, plus PIT calibrators for the forecaster.

    Rows in fold k are scored with CPS fitted on ``fit_sets[k]``: test ('T') on all calibration rows, calibration
    half 'A' on half 'B' and vice versa. Test outcomes are never in a fitting set.
    """
    y = base["y"].to_numpy(float)
    point = base["regime"].to_numpy(float)
    resid = y - point
    fit_sets = {"T": folds != "T", "A": folds == "B", "B": folds == "A"}
    n = len(base)
    res = {k: {"crps": np.empty(n), "lo": np.empty(n), "hi": np.empty(n), "pit": np.empty(n)}
           for k in ("cps_split", "cps_mondrian", "cps_pit", "cps_pit_mondrian")}
    fitted = {}
    for fold, fit in fit_sets.items():
        rows = folds == fold
        split = SplitCPS(resid[fit])
        mond = MondrianCPS(resid[fit], cat[fit])
        res["cps_split"]["crps"][rows] = split.crps(y[rows], point[rows])
        res["cps_split"]["pit"][rows] = split.cdf(y[rows], point[rows])
        res["cps_split"]["lo"][rows], res["cps_split"]["hi"][rows] = split.interval(point[rows])
        res["cps_mondrian"]["crps"][rows] = mond.crps(y[rows], point[rows], cat[rows])
        res["cps_mondrian"]["pit"][rows] = mond.cdf(y[rows], point[rows], cat[rows])
        res["cps_mondrian"]["lo"][rows], res["cps_mondrian"]["hi"][rows] = mond.interval(point[rows], cat[rows])
        pits = base["fc_pit"].to_numpy(float)
        fitted[fold] = {"split": split, "mondrian": mond, "pit": PITCalibrator(pits[fit]),
                        "pit_by_cat": {c: PITCalibrator(pits[fit & (cat == c)]) for c in np.unique(cat[fit]) if c >= 0}}

    return res, fitted


def run(base: pd.DataFrame, cfg: dict) -> dict:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    from scripts import spike_forecaster as sf

    spike, floor = cfg["spike"], cfg["floor"]
    folds = fold_labels(base)
    y = base["y"].to_numpy(float)
    point = base["regime"].to_numpy(float)
    resid = y - point
    raw_cat = risk_category(base["eligible"], base["p_up"], base["last"], spike, floor)
    mapping = merge_small(raw_cat[folds != "T"])
    cat = np.vectorize(mapping.get)(raw_cat)
    fit_sets = {"T": folds != "T", "A": folds == "B", "B": folds == "A"}  # rows in fold k are scored with fit_sets[k]

    res, fitted = residual_cps(base, cat, folds)
    res["cps_mondrian_sliding"] = sliding_mondrian(base, cat)
    n = len(base)

    # PIT-recalibrated forecaster: needs the forecaster's mixture atoms again
    model = sf.SpikeForecaster.load()
    frame = sf.build_frame(sf.read_panel(), cfg)
    model.set_history(frame)
    prepared, history, positions = model._prepare(frame.loc[base.index])
    for start in range(0, n, 2048):
        idx = np.arange(start, min(start + 2048, n))
        samples, weights = model._mixture(prepared, history, positions, idx)
        order = np.argsort(samples, axis=1)
        x, w = np.take_along_axis(samples, order, axis=1), np.take_along_axis(weights, order, axis=1)
        c = np.cumsum(w, axis=1) / w.sum(axis=1, keepdims=True)
        f_y = base["fc_pit"].to_numpy(float)[idx]
        for key in ("cps_pit", "cps_pit_mondrian"):
            crps, lo, hi, pit = (np.empty(len(idx)) for _ in range(4))
            for fold in np.unique(folds[idx]):
                for cc in (np.unique(cat[idx]) if key == "cps_pit_mondrian" else [None]):
                    m = (folds[idx] == fold) & ((cat[idx] == cc) if cc is not None else True)
                    if not m.any():
                        continue
                    fit = fitted[fold]
                    cal = fit["pit"] if cc is None else fit["pit_by_cat"].get(cc, fit["pit"])
                    new_w = recalibrated_weights(c[m], cal)
                    crps[m] = so.weighted_crps(x[m], np.maximum(new_w, 0), y[idx][m])
                    u_lo, u_hi = cal.levels()
                    lo[m] = quantile_rows(x[m], c[m], np.full(m.sum(), u_lo))
                    hi[m] = quantile_rows(x[m], c[m], np.full(m.sum(), u_hi))
                    pit[m] = cal.cdf(f_y[m])
            res[key]["crps"][idx], res[key]["lo"][idx], res[key]["hi"][idx], res[key]["pit"][idx] = crps, lo, hi, pit

    # isotonic recalibration of P(spike up) on calibration (cross-fitted on calibration, full fit for test)
    p_iso = np.empty(n)
    elig = base["eligible"].to_numpy(bool)
    up = base["up"].to_numpy(bool)
    p_raw = base["p_up"].to_numpy(float)
    for fold, fit in fit_sets.items():
        rows = folds == fold
        f = fit & elig
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_raw[f], up[f].astype(float))
        p_iso[rows] = iso.predict(p_raw[rows])
        if fold == "T":
            iso_full = iso

    scored = scored_mask(base)
    tables, iso_rows, rel_rows = [], [], []
    for split in ("calibration", "test"):
        m = scored & base["split"].eq(split).to_numpy()
        rows = base.loc[m]
        entries = [
            ("persistence", rows["persistence_crps"], rows["persistence_lo"], rows["persistence_hi"], rows["persistence"]),
            ("lightgbm_conformal", rows["lightgbm_crps"], rows["lightgbm_lo"], rows["lightgbm_hi"], rows["lightgbm"]),
            ("regime_switch", rows["regime_crps"], rows["regime_lo"], rows["regime_hi"], rows["regime"]),
            ("spike_forecaster", rows["fc_crps"], rows["fc_q05"], rows["fc_q95"], rows["regime"]),
        ]
        for key in res:
            entries.append((key, res[key]["crps"][m], res[key]["lo"][m], res[key]["hi"][m],
                            rows["regime"] if key in ("cps_split", "cps_mondrian") else rows["regime"]))
        for name, crps, lo, hi, pt in entries:
            for r in score_block(name, rows, np.asarray(crps, float), np.asarray(lo, float), np.asarray(hi, float), np.asarray(pt, float)):
                r["split"] = split
                tables.append(r)
        e = elig[m]
        for label, p in (("raw", p_raw[m]), ("isotonic", p_iso[m])):
            t, pp = up[m][e], p[e]
            iso_rows.append({"split": split, "probability": label, "n": int(e.sum()), "positives": int(t.sum()),
                             "brier": float(brier_score_loss(t, pp)), "log_loss": float(log_loss(t, np.clip(pp, 1e-6, 1 - 1e-6))),
                             "average_precision": float(average_precision_score(t, pp)), "mean_p": float(pp.mean()),
                             "base_rate": float(t.mean())})
            b = pd.cut(pp, [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], include_lowest=True)
            g = pd.DataFrame({"p": pp, "hit": t.astype(float), "bin": b}).groupby("bin", observed=True)
            rel = g.agg(mean_p=("p", "mean"), freq=("hit", "mean"), n=("hit", "size")).reset_index()
            rel.insert(0, "probability", label)
            rel.insert(0, "split", split)
            rel_rows.append(rel)
    master = pd.DataFrame(tables)
    # test bootstrap: CPS variants against the spike forecaster and persistence
    test = base.loc[scored & base["split"].eq("test").to_numpy()]
    tm = (scored & base["split"].eq("test").to_numpy())
    boot = []
    for key in res:
        for bname, bcrps in (("spike_forecaster", test["fc_crps"].to_numpy()), ("persistence", test["persistence_crps"].to_numpy())):
            for subset in ("all", "tail", "onset", "fresh_onset"):
                mask = np.ones(len(test), bool) if subset == "all" else test[subset].to_numpy(bool)
                mean, lo, hi = so.block_bootstrap_mean_diff(res[key]["crps"][tm] - bcrps, mask)
                boot.append({"model": key, "baseline": bname, "subset": subset, "metric": "crps", "diff": mean,
                             "boot_p05": lo, "boot_p95": hi, "n": int(mask.sum())})
    for bname, bcrps in (("persistence", test["persistence_crps"].to_numpy()),):
        for subset in ("all", "tail", "onset", "fresh_onset"):
            mask = np.ones(len(test), bool) if subset == "all" else test[subset].to_numpy(bool)
            mean, lo, hi = so.block_bootstrap_mean_diff(test["fc_crps"].to_numpy() - bcrps, mask)
            boot.append({"model": "spike_forecaster", "baseline": bname, "subset": subset, "metric": "crps", "diff": mean,
                         "boot_p05": lo, "boot_p95": hi, "n": int(mask.sum())})
    # per-row test outputs for figures
    rows_out = test[["y", "regime", "persistence", "lightgbm", "eligible", "tail", "onset", "p_up", "fc_q05", "fc_q95",
                     "fc_pit", "regime_lo", "regime_hi", "lightgbm_lo", "lightgbm_hi", "lightgbm_pit", "regime_pit",
                     "persistence_pit"]].copy()
    rows_out["p_up_isotonic"] = p_iso[tm]
    rows_out["risk_category"] = cat[tm]
    for key in res:
        for k in ("lo", "hi", "pit", "crps"):
            rows_out[f"{key}_{k}"] = res[key][k][tm]
    cats = pd.DataFrame({"category": raw_cat[folds != "T"]}).value_counts().rename("calibration_rows").reset_index()
    cats["merged_into"] = cats["category"].map(mapping)
    return {"master": master, "boot": pd.DataFrame(boot), "isotonic": pd.DataFrame(iso_rows),
            "reliability": pd.concat(rel_rows, ignore_index=True), "rows": rows_out, "categories": cats,
            "iso_thresholds": {"x": iso_full.X_thresholds_.tolist(), "y": iso_full.y_thresholds_.tolist()}}


def main():
    if BASE_CACHE.exists() and "--rebuild" not in sys.argv:
        from scripts.spike_forecaster import load_config

        base, cfg = pd.read_parquet(BASE_CACHE), load_config()
    else:
        base, cfg = build_base()
    print("base rows", len(base), flush=True)
    out = run(base, cfg)
    out["master"].to_csv(OUT / "final_cps_metrics.csv", index=False)
    out["boot"].to_csv(OUT / "final_cps_bootstrap.csv", index=False)
    out["isotonic"].to_csv(OUT / "final_isotonic.csv", index=False)
    out["reliability"].astype({"bin": str}).to_csv(OUT / "final_reliability.csv", index=False)
    out["categories"].to_csv(OUT / "final_cps_categories.csv", index=False)
    out["rows"].to_parquet(OUT / "final_cps_test_rows.parquet")
    (OUT / "final_isotonic_map.json").write_text(json.dumps(out["iso_thresholds"]))
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(out["master"].round(3).to_string(index=False))
        print(out["isotonic"].round(4).to_string(index=False))
        print(out["categories"].to_string(index=False))


if __name__ == "__main__":
    main()
