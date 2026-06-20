"""
compute_accuracy.py — stage 4 of the shelf-screener feedback loop (precision gate).

Reads review_batches/<date>/labels.csv (filled in by a human reviewer),
computes precision for that batch, and writes review_batches/<date>/accuracy.json.

  precision = count(human_label == Yes) / count(human_label in {Yes, No})

This is a precision metric only — it judges what the screeners *did* fire,
not what they missed (design doc section 3.4). The gate is precision >= 80%
("8 out of every 10 should have appeared"). Unlabeled rows are excluded from
the denominator rather than counted as failures, since a partially-reviewed
batch shouldn't look worse than it is.

Also walks back through prior batches to report a consecutive-pass streak,
since the design doc recommends requiring 2-3 consecutive passing batches
before treating a screener version as "tuned" (one good batch can be luck).

Usage:
  python pipeline/compute_accuracy.py --date 2026-06-19
  python pipeline/compute_accuracy.py                  # most recent labeled batch
"""

import argparse
import csv
import glob
import json
import os
import sys

GATE_THRESHOLD = 0.80
YES_VALUES = {"yes", "y", "true", "1"}
NO_VALUES = {"no", "n", "false", "0"}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm(label):
    return (label or "").strip().lower()


def score_labels(labels_path):
    """(yes, no, unlabeled, total, per_screener) from a labels.csv."""
    yes = no = unlabeled = total = 0
    per_screener = {}
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            screener = row.get("screener", "unknown")
            bucket = per_screener.setdefault(screener, {"yes": 0, "no": 0, "unlabeled": 0})
            v = _norm(row.get("human_label"))
            if v in YES_VALUES:
                yes += 1
                bucket["yes"] += 1
            elif v in NO_VALUES:
                no += 1
                bucket["no"] += 1
            else:
                unlabeled += 1
                bucket["unlabeled"] += 1
    return yes, no, unlabeled, total, per_screener


def precision_of(yes, no):
    denom = yes + no
    return (yes / denom) if denom > 0 else None


def find_most_recent_batch(batch_root):
    candidates = sorted(glob.glob(os.path.join(batch_root, "*", "labels.csv")))
    return os.path.dirname(candidates[-1]) if candidates else None


def consecutive_pass_streak(batch_root, before_date):
    """Count consecutive prior batches (most recent first, < before_date) that
    already have an accuracy.json with gate_pass == true. Stops at the first
    batch that fails, is unlabeled, or has no accuracy.json yet."""
    dates = sorted(
        d for d in os.listdir(batch_root)
        if os.path.isdir(os.path.join(batch_root, d)) and d < before_date
    )
    streak = 0
    for d in reversed(dates):
        acc_path = os.path.join(batch_root, d, "accuracy.json")
        if not os.path.exists(acc_path):
            break
        with open(acc_path) as f:
            acc = json.load(f)
        if acc.get("gate_pass") is True:
            streak += 1
        else:
            break
    return streak


def emit_github_output(gate_pass, precision, batch_date):
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        f.write(f"gate_pass={'true' if gate_pass else 'false'}\n")
        f.write(f"precision={precision if precision is not None else ''}\n")
        f.write(f"batch_date={batch_date}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None, help="Batch date (YYYY-MM-DD); default = most recent")
    ap.add_argument("--batch-root", default=os.path.join(REPO_ROOT, "review_batches"))
    args = ap.parse_args()

    if args.date:
        batch_dir = os.path.join(args.batch_root, args.date)
        labels_path = os.path.join(batch_dir, "labels.csv")
        if not os.path.exists(labels_path):
            sys.exit(f"No labels.csv for batch {args.date} (looked in {labels_path})")
    else:
        batch_dir = find_most_recent_batch(args.batch_root)
        if batch_dir is None:
            sys.exit(f"No batches with labels.csv found under {args.batch_root}")
        labels_path = os.path.join(batch_dir, "labels.csv")

    batch_date = os.path.basename(batch_dir)
    yes, no, unlabeled, total, per_screener = score_labels(labels_path)
    precision = precision_of(yes, no)
    gate_pass = (precision is not None) and (precision >= GATE_THRESHOLD)
    streak = consecutive_pass_streak(args.batch_root, batch_date)
    streak_after = streak + 1 if gate_pass else 0

    accuracy = {
        "batch_date": batch_date,
        "total_signals": total,
        "labeled": yes + no,
        "unlabeled": unlabeled,
        "yes": yes,
        "no": no,
        "precision": round(precision, 4) if precision is not None else None,
        "gate_threshold": GATE_THRESHOLD,
        "gate_pass": gate_pass,
        "consecutive_passes": streak_after,
        "per_screener": per_screener,
    }

    out_path = os.path.join(batch_dir, "accuracy.json")
    with open(out_path, "w") as f:
        json.dump(accuracy, f, indent=2)

    print(f"=== Batch {batch_date} ===")
    print(f"  labeled {yes + no}/{total} ({unlabeled} unlabeled)")
    if precision is None:
        print("  precision: n/a (no labeled rows yet)")
    else:
        verdict = "PASS" if gate_pass else "FAIL"
        print(f"  precision: {precision:.1%}  (gate >= {GATE_THRESHOLD:.0%})  -> {verdict}")
        print(f"  consecutive passing batches (incl. this one): {streak_after}")
    for screener, b in per_screener.items():
        p = precision_of(b["yes"], b["no"])
        p_str = f"{p:.1%}" if p is not None else "n/a"
        print(f"    {screener}: yes={b['yes']} no={b['no']} unlabeled={b['unlabeled']} precision={p_str}")
    print(f"\nWrote {out_path}")

    emit_github_output(gate_pass, precision, batch_date)


if __name__ == "__main__":
    main()
