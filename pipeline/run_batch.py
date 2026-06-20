"""
run_batch.py — stage 1 of the shelf-screener feedback loop.

Runs the three shelf screeners (HOLD, FLEX, STRICT) on one shared yfinance
download and writes a review batch:

  review_batches/<date>/results.json   full gate-metric dict per signal
  review_batches/<date>/labels.csv     trimmed sheet for the human reviewer

No LLM calls happen here — this is the off-LLM scheduled scan (design doc
section 3.1). Charts are generated separately by chart_gen.py so this stage
stays fast and network-light (one shared download for the whole universe).

Usage:
  python pipeline/run_batch.py                     # as of today
  python pipeline/run_batch.py --date 2026-06-19    # as of a past date
  python pipeline/run_batch.py --max-symbols 50     # smoke test on a subset
"""

import argparse
import csv
import json
import math
import os
import sys
import datetime as dt

import pandas as pd

# Make `screeners/` importable regardless of cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from screeners import shelf_hold_screener as hold_mod          # noqa: E402
from screeners import shelf_reclaim_screener as flex_mod       # noqa: E402
from screeners import shelf_reclaim_screener2 as strict_mod    # noqa: E402

SCREENERS = [
    ("HOLD",   hold_mod),
    ("FLEX",   flex_mod),
    ("STRICT", strict_mod),
]

# Columns shown to the human reviewer (kept short on purpose; results.json
# has the full gate-metric dict for anyone who wants more detail).
LABEL_COLUMNS = [
    "batch_date", "symbol", "screener", "signal_date", "close", "shelf",
    "shelf_touches", "atr_pct", "ma50_gap_pct", "score", "image_path",
    "human_label", "reviewer", "comment", "reviewed_at",
]


def _json_default(o):
    """Make pandas/numpy values JSON-safe."""
    if isinstance(o, float) and math.isnan(o):
        return None
    if pd.isna(o):
        return None
    if hasattr(o, "item"):   # numpy scalar
        return o.item()
    return str(o)


def _clean_record(rec):
    """Replace NaN/NaT with None so json.dumps doesn't emit invalid NaN tokens."""
    out = {}
    for k, v in rec.items():
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
                out[k] = None
                continue
        except (TypeError, ValueError):
            pass
        out[k] = v.item() if hasattr(v, "item") else v
    return out


def run_batch(as_of, out_root, max_symbols=None, no_data_cache=False):
    as_of = pd.Timestamp(as_of)
    batch_date = str(as_of.date())
    out_dir = os.path.join(out_root, batch_date)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Shelf-screener batch run as of {batch_date} ===")
    print("Loading universe (Midcap 150 + Smallcap 250 + Microcap 250)...")
    universe = hold_mod.get_universe()
    if not universe:
        sys.exit("No symbols loaded - aborting.")
    symbols = sorted(universe)
    if max_symbols:
        symbols = symbols[:max_symbols]
        universe = {s: universe[s] for s in symbols}
    print(f"  {len(symbols)} symbols")

    history_days = max(m.HISTORY_DAYS for _, m in SCREENERS)
    start = (as_of - pd.Timedelta(days=history_days)).date()
    end = (as_of + pd.Timedelta(days=1)).date()

    import yfinance as yf
    print(f"Downloading daily data {start} -> {as_of.date()} (one shot, {len(symbols)} symbols)...")
    data = yf.download([s + ".NS" for s in symbols], start=str(start), end=str(end),
                        interval="1d", group_by="ticker", auto_adjust=True,
                        threads=True, progress=True)

    all_records = []
    for label, mod in SCREENERS:
        print(f"\n--- Running {label} ---")
        try:
            out = mod.run_scan(as_of, data=data, universe=universe)
        except Exception as e:
            print(f"  {label} FAILED: {e}")
            continue
        if out is None or out.empty:
            print(f"  {label}: 0 signals")
            continue
        print(f"  {label}: {len(out)} signal(s)")
        for rec in out.to_dict("records"):
            rec = _clean_record(rec)
            rec["screener"] = label
            rec["batch_date"] = batch_date
            rec["signal_date"] = rec.get("date")
            rec["image_path"] = f"charts/{rec['symbol']}_{label}.png"
            all_records.append(rec)

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_records, f, indent=2, default=_json_default)
    print(f"\nWrote {len(all_records)} signal(s) to {results_path}")

    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for rec in all_records:
            row = {col: rec.get(col, "") for col in LABEL_COLUMNS}
            row["human_label"] = ""
            row["reviewer"] = ""
            row["comment"] = ""
            row["reviewed_at"] = ""
            writer.writerow(row)
    print(f"Wrote review sheet to {labels_path}")

    return results_path, labels_path, len(all_records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None, help="As-of date (YYYY-MM-DD); default = today")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "review_batches"))
    ap.add_argument("--max-symbols", type=int, default=None,
                     help="Limit universe size (smoke testing / CI time control)")
    args = ap.parse_args()

    as_of = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    run_batch(as_of, args.out_dir, max_symbols=args.max_symbols)


if __name__ == "__main__":
    main()
