"""Extract system-level fields from AEMO WEM Reference pre-dispatch runs.

Source: https://data.wa.aemo.com.au/public/market-data/wemde/dispatchSolution/
preDispatchData/previous/DispatchSolutionPreDispatch_YYYYMMDD.zip, one ZIP per
trading day (08:00 to 07:30 next day, AWST) with 48 half-hourly run files named
ReferencePre-DispatchSolution_YYYYMMDDHHMM.json. Each run holds 96 future
5-minute dispatch intervals.

The full archive is ~160-250 MB per day and the server serves ~0.25 MB/s per
connection, so only selected runs are read with HTTP range requests (remotezip)
and only system-level fields are kept. ``--pass 1`` takes runs labelled at even
hours (:00), ``--pass 2`` the odd hours. The 75 MB run JSON is not stored.

Availability: the public site posts each run file about 35 minutes after its
label (checked on the live ``current`` folder, 2026-10-08). The rule used
downstream is available_at = max(label + 40 min, issue time + 25 min), where
issue time is parsed from ``dispatchDataIssueID``. Times are AWST (UTC+8).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "aemo_predispatch"
BASE = "https://data.wa.aemo.com.au/public/market-data/wemde/dispatchSolution/preDispatchData/previous/"

BLOCKS = {
    "prices": ("energy", "regulationRaise", "regulationLower", "contingencyRaise", "contingencyLower"),
    "marketServiceRequirements": ("energy", "contingencyRaise", "regulationRaise"),
    "inServiceQuantities": ("energyInjectionCapacity",),
    "availableQuantities": ("energyInjectionCapacity", "contingencyRaise", "regulationRaise"),
    "dispatchTotal": ("energyInjectionCapacity", "energyWithdrawalCapacity"),
    "marketShortfalls": ("energyDeficit", "contingencyRaiseDeficit", "regulationRaiseDeficit"),
    "contingencySolution": ("demandLevel", "dpvLevel", "largestContingency"),
}
RE_ISSUE = re.compile(rb'"dispatchDataIssueID":"(\d{17})')


OVERLAP = 8192
CHUNK = 8 * 1024 * 1024
KEYS = {"dispatch_interval": (b'"dispatchInterval":"', b'"'), "solve_status": (b'"solveStatus":"', b'"')}
KEYS.update({name: (b'"' + name.encode() + b'":{', b"}") for name in BLOCKS})


def _scan(buffer: bytes, key: bytes, stop: bytes, limit: int) -> list[bytes]:
    """Values after each ``key`` (up to ``stop``) whose key starts before ``limit``."""
    out, pos = [], 0
    while True:
        i = buffer.find(key, pos)
        if i < 0 or i >= limit:
            return out
        j = buffer.find(stop, i + len(key))
        out.append(buffer[i + len(key):j])
        pos = j


def parse_stream(stream, label: str) -> pd.DataFrame:
    """Parse a run file chunk by chunk so a 120 MB JSON never sits in memory.

    A key is taken from a buffer only if it starts more than OVERLAP bytes
    before the buffer end; the tail is carried into the next buffer, so every
    match is counted exactly once (each value is a few hundred bytes).
    """
    found: dict[str, list[bytes]] = {k: [] for k in KEYS}
    carry, head, issue = b"", b"", None
    while True:
        chunk = stream.read(CHUNK)
        buffer = carry + chunk
        if not head:
            head = buffer[:4000]
            m = RE_ISSUE.search(head)
            issue = m.group(1).decode()[:14] if m else None
        limit = len(buffer) if not chunk else len(buffer) - OVERLAP
        for name, (key, stop) in KEYS.items():
            found[name].extend(_scan(buffer, key, stop, limit))
        if not chunk:
            break
        carry = buffer[limit:]
    intervals = [v.decode() for v in found["dispatch_interval"]]
    n = len(intervals)
    cols: dict[str, list] = {"dispatch_interval": intervals}
    solve = [v.decode() for v in found["solve_status"]]
    cols["solve_status"] = solve if len(solve) == n else [None] * n
    for name, keys in BLOCKS.items():
        blocks = [json.loads(b"{" + v + b"}") for v in found[name]]
        if len(blocks) != n:
            blocks = [{}] * n
        for key in keys:
            cols[f"{name}.{key}"] = [blk.get(key) for blk in blocks]
    frame = pd.DataFrame(cols)
    frame.insert(0, "run_label", label)
    frame.insert(1, "issue_id_time", issue)
    return frame


def parse_run(raw: bytes, label: str) -> pd.DataFrame:
    return parse_stream(io.BytesIO(raw), label)


def run_names(day: pd.Timestamp, which: int) -> list[str]:
    start = day + pd.Timedelta(hours=8)
    labels = pd.date_range(start, start + pd.Timedelta(hours=23, minutes=30), freq="30min")
    keep = [t for t in labels if t.minute == 0 and (t.hour % 2 == 0) == (which == 1)]
    return [f"ReferencePre-DispatchSolution_{t:%Y%m%d%H%M}.json" for t in keep]


def fetch_member(url: str, name: str) -> pd.DataFrame | None:
    err = None
    for attempt in range(4):
        try:
            with RemoteZip(url) as z:
                if name not in z.namelist():
                    return None
                with z.open(name) as stream:
                    return parse_stream(stream, name.split("_")[1][:12])
        except Exception as exc:  # network errors are retried, then logged
            err = exc
            time.sleep(10 * (attempt + 1))
    print(f"FAILED {url} {name}: {err}", file=sys.stderr, flush=True)
    return None


def fetch_day(day: pd.Timestamp, which: int, dest: Path, threads: int) -> str:
    url = BASE + f"DispatchSolutionPreDispatch_{day:%Y%m%d}.zip"
    names = run_names(day, which)
    with ThreadPoolExecutor(threads) as ex:
        parts = [p for p in ex.map(lambda n: fetch_member(url, n), names) if p is not None]
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(dest / f"{day:%Y%m%d}.parquet", index=False)
    return f"{day:%Y-%m-%d} runs={len(parts)}/{len(names)}"


CONSOLIDATED = OUT / "predispatch_runs_first9h.parquet"
KEEP_INTERVALS = 18  # 9 hours of half-hourly intervals per run; features read at most 6 h past the run label


def consolidate() -> None:
    """Merge the per-day extracts into one committed file, first 9 hours of each run."""
    files = sorted((OUT / "extract").glob("pass*/*.parquet"))
    runs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    runs = runs.drop_duplicates(["run_label", "dispatch_interval"]).sort_values(["run_label", "dispatch_interval"])
    runs = runs.groupby("run_label", group_keys=False).head(KEEP_INTERVALS)
    runs.to_parquet(CONSOLIDATED, index=False, compression="zstd")
    labels = pd.to_datetime(runs["run_label"], format="%Y%m%d%H%M")
    print(f"{CONSOLIDATED.name}: {runs['run_label'].nunique()} runs, {labels.min()} to {labels.max()}, {len(runs)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", type=int, default=1, choices=(1, 2))
    ap.add_argument("--start", default="2023-09-30")
    ap.add_argument("--end", default="2026-08-17")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--consolidate", action="store_true", help="only merge existing extracts")
    args = ap.parse_args()
    if args.consolidate:
        consolidate()
        return
    dest = OUT / "extract" / f"pass{args.which}"
    dest.mkdir(parents=True, exist_ok=True)
    days = [d for d in pd.date_range(args.start, args.end, freq="D") if not (dest / f"{d:%Y%m%d}.parquet").exists()]
    with ProcessPoolExecutor(args.processes) as ex:
        futures = [ex.submit(fetch_day, day, args.which, dest, args.threads) for day in days]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
