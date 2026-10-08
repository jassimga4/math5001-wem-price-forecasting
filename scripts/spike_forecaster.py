"""Implemented WEM 5-minute spike forecaster (frozen from calibration evidence).

For a target interval T the forecast is issued at origin T - 5 min. Everything
used is known by then: panel lags (scripts/forecast_design.py), the frozen
regime-switch point model, price-path summaries of prices labelled T - 5 min or
earlier, and the regime-switch residuals of earlier intervals whose prices had
been realised.

Per target the forecaster gives:

* ``point``: the regime switch (the hurdle point override is off; see
  configs/spike_forecaster.json for why);
* ``p_up`` / ``p_down``: probabilities that the price crosses the training 95th
  percentile upward or the 5th percentile downward, for origins whose last price
  is inside that band (NaN otherwise, where the regime switch already handles
  the tail);
* a predictive distribution: the regime switch's 7-day residual window around the
  point, with weight ``p`` moved onto the size models' quantile grid when a
  classifier passes its calibration-chosen mixture cutoff. Quantiles, CDF values
  at the spike and floor thresholds, and CRPS come from that mixture;
* ``alert``: the spike alert at the calibration-chosen detection cutoff.

Usage::

    python scripts/spike_forecaster.py fit        # fit on train, cutoffs on calibration, save
    python scripts/spike_forecaster.py evaluate   # test-window forecasts, metrics, validation
    python scripts/spike_forecaster.py forecast --start 2026-08-17 --end 2026-08-18 --out f.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import spike_onset as so  # noqa: E402
from scripts.forecast_design import TARGET, TREE_FEATURES, build_exante_frame, split_mask  # noqa: E402
from scripts.metrics import crps_from_residual_samples  # noqa: E402
from scripts.paths import panel_path  # noqa: E402
from scripts.price_path import near_band, price_path_features  # noqa: E402
from scripts.regime_switch import MODEL_PATH as POINT_MODEL_PATH  # noqa: E402
from scripts.regime_switch import WINDOW, forecasts, mask_for, switched  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "spike_forecaster.json"
MODEL_PATH = ROOT / "models" / "spike_forecaster.joblib"
OUT = ROOT / "reports" / "forecast"
TEST_FORECASTS = OUT / "spike_forecaster_test_forecasts.parquet"
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
CHUNK = 2048


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- data


def read_panel(path: Path | None = None) -> pd.DataFrame:
    raw = pd.read_parquet(path or panel_path(), engine="pyarrow")
    for col in ("interval_end", "trading_interval_end"):
        if col in raw.columns:
            raw[col] = pd.to_datetime(raw[col]).astype("datetime64[ns]")
    return raw


def build_frame(raw: pd.DataFrame, config: dict, point_bundle: dict | None = None) -> pd.DataFrame:
    """One row per target interval whose origin-time inputs are complete.

    ``y`` (the target price) may be missing, as it is for a live origin; it is
    only read for scoring and, for earlier rows, as realised history.
    """
    exante = build_exante_frame(raw)
    need = [*TREE_FEATURES, "forecast_origin"]
    base = exante.loc[exante[need].notna().all(axis=1), [TARGET, *need]].sort_index()
    saved = point_bundle or joblib.load(POINT_MODEL_PATH)
    frame = forecasts(base, saved["bundle"], saved["features"])
    rule = config["regime_rule"]
    frame["regime"] = switched(frame, mask_for(frame, (rule["rule"], rule["feature"], rule["threshold"],
                                                       config["spike"], config["floor"])))
    frame = frame.join(base[TREE_FEATURES]).join(base["forecast_origin"])
    frame["split"] = split_mask(frame.index)
    spike, floor = config["spike"], config["floor"]
    frame["eligible"] = (frame["last"] < spike) & (frame["last"] > floor)
    band = config["near_band"]
    realised = frame["y"].dropna()  # a price is used for row T only through shift(k >= 1)
    frame = frame.join(price_path_features(realised, frame.index, band["near_up"], band["near_down"]))
    return add_labels(frame, realised, spike, floor)


def add_labels(frame: pd.DataFrame, realised: pd.Series, spike: float, floor: float) -> pd.DataFrame:
    """Scoring labels (need the realised target); NaN-safe: rows without y get False."""
    y = frame["y"]
    frame["tail"] = (y >= spike) | (y <= floor)
    frame["up"] = frame["eligible"] & (y >= spike)
    frame["down"] = frame["eligible"] & (y <= floor)
    frame["onset"] = frame["eligible"] & frame["tail"]
    frame["rest_of_event"] = ~frame["eligible"] & frame["tail"]
    grid = realised.reindex(pd.date_range(realised.index.min(), realised.index.max(), freq="5min"))
    grid_tail = ((grid >= spike) | (grid <= floor)).astype(float).where(grid.notna())
    recent = pd.concat([grid_tail.shift(k) for k in range(1, 13)], axis=1).max(axis=1, skipna=True)
    frame["fresh_onset"] = frame["onset"] & recent.reindex(frame.index).eq(0).to_numpy()
    return frame


# --------------------------------------------------------------------------- models


def make_classifier(kind: str, params: dict):
    if kind in ("lightgbm", "lightgbm_wide"):
        return LGBMClassifier(**{**so.CLF_PARAMS, **params})
    if kind in ("hist_gb", "hist_gb_wide"):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(**{"min_samples_leaf": 200, "l2_regularization": 1.0,
                                                 "early_stopping": False, "random_state": 7, **params})
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**{**so.XGB_PARAMS, **params})
    raise ValueError(f"unsupported frozen classifier: {kind}")


def weighted_quantiles(samples: np.ndarray, weights: np.ndarray, levels) -> np.ndarray:
    """Left-continuous inverse of a weighted empirical CDF, one row per origin."""
    order = np.argsort(samples, axis=1)
    x = np.take_along_axis(samples, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    cdf = np.cumsum(w, axis=1) / w.sum(axis=1, keepdims=True)
    out = np.empty((len(x), len(levels)))
    for j, q in enumerate(levels):
        idx = np.minimum((cdf < q - 1e-12).sum(axis=1), x.shape[1] - 1)
        out[:, j] = x[np.arange(len(x)), idx]
    return out


class SpikeForecaster:
    """Hurdle spike forecaster around the regime switch. Fit once; forecast any origin."""

    def __init__(self, config: dict):
        self.config = config
        self.features = list(config["features"])
        self.models: dict = {}
        self.cutoffs: dict = dict(config.get("cutoffs", {}))
        self.history: pd.Series | None = None  # realised regime-switch residuals by target time

    # ---------------------------------------------------------------- fitting
    def fit(self, frame: pd.DataFrame, select_cutoffs: bool = True) -> "SpikeForecaster":
        cfg = self.config
        train = frame.loc[frame["split"].eq("train") & frame["y"].notna()]
        fit_start = pd.Timestamp(cfg["windows"]["fit_start"])
        rows = train.loc[train["eligible"] & (train.index >= fit_start)]
        clf = cfg["classifier"]
        for side in ("up", "down"):
            model = make_classifier(clf["kind"], {**clf.get("fixed", {}), **clf["params"][side]})
            model.fit(so._xy(rows, self.features), rows[side].astype(int))
            self.models[f"clf_{side}"] = model
            self.models[f"size_{side}"], _ = so.fit_size(train, self.features, side)
        self.set_history(frame)
        if select_cutoffs:
            self.cutoffs = self.select_cutoffs(frame)
        return self

    def set_history(self, frame: pd.DataFrame) -> None:
        realised = frame.loc[frame["y"].notna()]
        self.history = (realised["y"] - realised["regime"]).astype(float)

    def select_cutoffs(self, frame: pd.DataFrame) -> dict:
        """Calibration-only cutoffs, same rules and grids as scripts/spike_onset.py."""
        cal = self.evaluation_block(frame, "calibration")
        prepared, history, positions = self._prepare(cal)
        plain = self._plain_crps(prepared, history, positions)
        base_crps = float(plain.mean())
        tail = prepared["tail"].to_numpy()
        grid = []
        for c_up in so.MIX_CUTOFFS:
            for c_dn in so.MIX_CUTOFFS:
                pi_u, pi_d = so.mixture_weights(prepared, c_up, c_dn)
                crps = so.mixture_crps(prepared, history, positions, pi_u, pi_d, plain)
                grid.append({"c_up": c_up, "c_down": c_dn, "cal_crps": float(crps.mean()),
                             "cal_tail_crps": float(crps[tail].mean())})
        grid = pd.DataFrame(grid)
        ok = grid.loc[grid["cal_crps"] <= base_crps * 1.01]
        best = (ok if not ok.empty else grid).sort_values(["cal_tail_crps", "cal_crps"]).iloc[0]
        p_any = np.maximum(prepared["p_up"], prepared["p_down"]).to_numpy()
        f1s = []
        for c in np.round(np.arange(0.02, 0.96, 0.02), 2):
            d = so.detection(p_any >= c, prepared)
            prec, rec = d["precision"], d["recall"]
            f1s.append((2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) else 0.0, c))
        f1, c_det = max(f1s)
        return {"mixture_up": float(best["c_up"]), "mixture_down": float(best["c_down"]),
                "detection": float(c_det), "calibration_detection_f1": float(f1)}

    # ---------------------------------------------------------------- forecasting
    def _prepare(self, block: pd.DataFrame):
        out = block.copy()
        x = so._xy(out, self.features)
        out["p_up"] = self.models["clf_up"].predict_proba(x)[:, 1]
        out["p_down"] = self.models["clf_down"].predict_proba(x)[:, 1]
        out["up_median"], up_grid = so.predict_size(self.models["size_up"], out, self.features)
        out["down_median"], down_grid = so.predict_size(self.models["size_down"], out, self.features)
        out["up_grid"], out["down_grid"] = list(up_grid), list(down_grid)
        hist_index = self.history.index.to_numpy()
        positions = np.searchsorted(hist_index, out.index.to_numpy(), side="left")  # strictly earlier rows only
        if (positions < WINDOW).any():
            raise ValueError(f"{int((positions < WINDOW).sum())} origins have fewer than {WINDOW} earlier realised residuals")
        return out, self.history.to_numpy(float), positions

    def _plain_crps(self, prepared, history, positions) -> np.ndarray:
        res = so.residual_samples(history, positions)
        return crps_from_residual_samples(res, prepared["y"].to_numpy(float), prepared["regime"].to_numpy(float))

    def _weights(self, prepared):
        return so.mixture_weights(prepared, self.cutoffs["mixture_up"], self.cutoffs["mixture_down"])

    def _mixture(self, prepared, history, positions, rows):
        """Samples and weights of the predictive mixture for the given row positions."""
        pi_u, pi_d = self._weights(prepared)
        centre = prepared["regime"].to_numpy(float)[rows]
        base = centre[:, None] + so.residual_samples(history, positions[rows])
        up = np.vstack(prepared["up_grid"].to_numpy()[rows])
        down = np.vstack(prepared["down_grid"].to_numpy()[rows])
        k = up.shape[1]
        samples = np.hstack([base, up, down])
        weights = np.hstack([np.repeat(((1 - pi_u[rows] - pi_d[rows]) / WINDOW)[:, None], WINDOW, axis=1),
                             np.repeat((pi_u[rows] / k)[:, None], k, axis=1),
                             np.repeat((pi_d[rows] / k)[:, None], k, axis=1)])
        return samples, weights

    def forecast(self, frame: pd.DataFrame, targets=None, levels=QUANTILES) -> pd.DataFrame:
        """Forecasts for target intervals (index labels); origins are targets - 5 min.

        Uses only columns built from data available at each origin, plus residuals
        of earlier targets in ``self.history``. Call ``set_history(frame)`` first if
        ``frame`` has newer realised prices than the fitted history.
        """
        block = frame if targets is None else frame.loc[pd.DatetimeIndex(targets)]
        prepared, history, positions = self._prepare(block)
        pi_u, pi_d = self._weights(prepared)
        elig = prepared["eligible"].to_numpy()
        spike, floor = self.config["spike"], self.config["floor"]
        q = np.empty((len(prepared), len(levels)))
        above, below = np.empty(len(prepared)), np.empty(len(prepared))
        for start in range(0, len(prepared), CHUNK):
            rows = np.arange(start, min(start + CHUNK, len(prepared)))
            samples, weights = self._mixture(prepared, history, positions, rows)
            q[rows] = weighted_quantiles(samples, weights, levels)
            total = weights.sum(axis=1)
            above[rows] = (weights * (samples >= spike)).sum(axis=1) / total
            below[rows] = (weights * (samples <= floor)).sum(axis=1) / total
        p_up = np.where(elig, prepared["p_up"], np.nan)
        p_down = np.where(elig, prepared["p_down"], np.nan)
        out = pd.DataFrame({
            "origin": prepared.index - pd.Timedelta(minutes=5),
            "point": prepared["regime"].to_numpy(float),
            "p_up": p_up,
            "p_down": p_down,
            "eligible": elig,
            "alert": elig & (np.maximum(prepared["p_up"], prepared["p_down"]).to_numpy() >= self.cutoffs["detection"]),
            "mix_weight_up": pi_u,
            "mix_weight_down": pi_d,
            "prob_at_or_above_spike": above,
            "prob_at_or_below_floor": below,
        }, index=prepared.index)
        out.index.name = "target"
        for j, level in enumerate(levels):
            out[f"q{int(round(level * 100)):02d}"] = q[:, j]
        return out

    def crps(self, frame: pd.DataFrame) -> np.ndarray:
        """CRPS of the predictive mixture against realised y (rows must have y)."""
        prepared, history, positions = self._prepare(frame)
        pi_u, pi_d = self._weights(prepared)
        plain = self._plain_crps(prepared, history, positions)
        return so.mixture_crps(prepared, history, positions, pi_u, pi_d, plain)

    # ---------------------------------------------------------------- evaluation
    def evaluation_block(self, frame: pd.DataFrame, split: str) -> pd.DataFrame:
        """Scored rows of a split; calibration skips its first residual window, as in spike_onset."""
        rows = frame.loc[frame["split"].eq(split) & frame["y"].notna()]
        if split == "calibration":
            rows = rows.iloc[WINDOW:]
        return rows

    def metrics(self, frame: pd.DataFrame, fc: pd.DataFrame | None = None, crps: np.ndarray | None = None) -> dict:
        fc = self.forecast(frame) if fc is None else fc
        crps = self.crps(frame) if crps is None else crps
        called = fc["alert"].to_numpy()
        row = so.summary("spike_forecaster", frame, fc["point"].to_numpy(float), crps, called)
        elig = frame["eligible"].to_numpy()
        truth = frame["onset"].to_numpy()[elig]
        p_any = np.maximum(fc["p_up"], fc["p_down"]).to_numpy()[elig]
        row.update(onset_average_precision=float(average_precision_score(truth, p_any)) if truth.any() else np.nan,
                   onset_roc_auc=float(roc_auc_score(truth, p_any)) if truth.any() else np.nan)
        return row

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path = MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "models": self.models, "cutoffs": self.cutoffs, "history": self.history},
                    path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "SpikeForecaster":
        saved = joblib.load(path)
        obj = cls(saved["config"])
        obj.models, obj.cutoffs, obj.history = saved["models"], saved["cutoffs"], saved["history"]
        return obj


# --------------------------------------------------------------------------- CLI


def _baseline_crps(frame, block, column):
    realised = frame.loc[frame["y"].notna()]
    hist = (realised["y"] - realised[column]).to_numpy(float)
    pos = np.searchsorted(realised.index.to_numpy(), block.index.to_numpy(), side="left")
    return crps_from_residual_samples(so.residual_samples(hist, pos), block["y"].to_numpy(float),
                                      block[column].to_numpy(float))


def cmd_fit(args):
    config = load_config(args.config)
    frame = build_frame(read_panel(), config)
    model = SpikeForecaster(config).fit(frame)
    frozen = config.get("cutoffs", {})
    mismatch = {k: (frozen.get(k), v) for k, v in model.cutoffs.items() if k in frozen and not np.isclose(frozen[k], v)}
    if mismatch:
        raise RuntimeError(f"calibration cutoffs differ from the frozen config: {mismatch}")
    print("cutoffs", model.cutoffs)
    print("saved", model.save(args.model))


def cmd_evaluate(args):
    model = SpikeForecaster.load(args.model)
    config = model.config
    frame = build_frame(read_panel(), config)
    model.set_history(frame)
    results, per_split = [], {}
    for split in ("calibration", "test"):
        block = model.evaluation_block(frame, split)
        fc = model.forecast(block)
        crps = model.crps(block)
        row = model.metrics(block, fc, crps)
        row["split"] = split
        results.append(row)
        per_split[split] = (block, fc, crps)
        if split == "test":
            fc.assign(y=block["y"], crps=crps).to_parquet(args.out)
    results = pd.DataFrame(results)
    results.to_csv(OUT / "spike_forecaster_metrics.csv", index=False)
    # bootstrap against persistence and the regime switch on both splits (1-day blocks, 90%)
    boot = []
    for split, (block, fc, crps) in per_split.items():
        y = block["y"].to_numpy(float)
        for base in ("persistence", "regime"):
            b_crps = _baseline_crps(frame, block, base)
            b_hat = block[base].to_numpy(float)
            for subset in so.BOOT_SUBSETS:
                mask = np.ones(len(y), bool) if subset == "all" else block[subset].to_numpy()
                for metric, a, b in (("mae", np.abs(y - fc["point"].to_numpy()), np.abs(y - b_hat)), ("crps", crps, b_crps)):
                    mean, lo, hi = so.block_bootstrap_mean_diff(a - b, mask)
                    boot.append({"split": split, "baseline": "regime_switch" if base == "regime" else base, "subset": subset,
                                 "metric": metric, "diff": mean, "boot_p05": lo, "boot_p95": hi, "n": int(mask.sum())})
    pd.DataFrame(boot).to_csv(OUT / "spike_forecaster_bootstrap.csv", index=False)
    # validation against the frozen comparison row of scripts/spike_onset.py
    ref = pd.read_csv(OUT / "spike_stage1_results.csv")
    ref = ref.loc[ref["model"].eq(config["comparison_row"])].set_index("split")
    keys = ["mae", "crps", "tail_mae", "tail_crps", "onset_mae", "onset_crps", "fresh_onset_mae", "fresh_onset_crps",
            "onset_precision", "onset_recall", "onset_average_precision", "onset_roc_auc"]
    check = {s: {k: float(r[k] - ref.loc[s, k]) for k in keys} for s, r in results.set_index("split").iterrows()}
    worst = max(abs(v) for d in check.values() for v in d.values())
    (OUT / "spike_forecaster_validation.json").write_text(json.dumps(
        {"comparison_row": config["comparison_row"], "max_abs_difference": worst, "differences": check}, indent=2))
    print(results[["split", *keys]].round(4).to_string(index=False))
    print("max abs difference against", config["comparison_row"], worst)
    if worst > 1e-6:
        raise SystemExit("module metrics do not reproduce the frozen comparison row")


def cmd_forecast(args):
    model = SpikeForecaster.load(args.model)
    frame = build_frame(read_panel(args.panel) if args.panel else read_panel(), model.config)
    model.set_history(frame)
    rows = frame.loc[(frame.index >= pd.Timestamp(args.start)) & (frame.index <= pd.Timestamp(args.end))]
    fc = model.forecast(rows)
    out = Path(args.out)
    fc.to_parquet(out) if out.suffix == ".parquet" else fc.to_csv(out)
    print(f"{len(fc)} forecasts -> {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fit").set_defaults(func=cmd_fit)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--out", default=TEST_FORECASTS)
    ev.set_defaults(func=cmd_evaluate)
    fc = sub.add_parser("forecast")
    fc.add_argument("--start", required=True)
    fc.add_argument("--end", required=True)
    fc.add_argument("--out", required=True)
    fc.add_argument("--panel", default=None, help="panel parquet (defaults to data/processed/wem_5min_panel.parquet)")
    fc.set_defaults(func=cmd_forecast)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
