"""
tools.py — helper functions used by the LangGraph debug agent (agent/debug_agent.py).

Grouped by what each LangGraph node needs:
  - feedback/code loading   : load_labeled_feedback, load_missed_signals, load_screener_source
  - patch application       : write_screener_source
  - regression validation   : validate_no_regression
  - git/PR plumbing          : create_branch, commit_and_push, open_pr

None of this calls an LLM directly — debug_agent.py owns the LangChain/LangGraph
wiring and prompt construction. This module is plain Python + subprocess (git/gh
CLI) + importlib, so it can be unit-tested without any LLM or network access
(aside from validate_no_regression's yfinance refetch).
"""

import csv
import glob
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCREENER_FILES = {
    "HOLD": "shelf_hold_screener.py",
    "FLEX": "shelf_reclaim_screener.py",
    "STRICT": "shelf_reclaim_screener2.py",
}


# ---------------------------------------------------------------------------
# Feedback + code loading
# ---------------------------------------------------------------------------

def load_screener_source(screener_name, repo_root=REPO_ROOT):
    path = os.path.join(repo_root, "screeners", SCREENER_FILES[screener_name])
    with open(path, "r") as f:
        return f.read()


def write_screener_source(screener_name, new_source, repo_root=REPO_ROOT):
    path = os.path.join(repo_root, "screeners", SCREENER_FILES[screener_name])
    with open(path, "w") as f:
        f.write(new_source)
    return path


def _all_batch_dirs(repo_root=REPO_ROOT):
    batch_root = os.path.join(repo_root, "review_batches")
    return sorted(
        d for d in glob.glob(os.path.join(batch_root, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "labels.csv"))
    )


def load_labeled_feedback(screener_name, repo_root=REPO_ROOT, max_batches=10):
    """False positives for one screener: rows where human_label == No, merged
    with the matching full record from that batch's results.json (so the agent
    sees every gate metric, not just the trimmed labels.csv columns).

    Returns a list of dicts: {batch_date, symbol, comment, record}.
    """
    false_positives = []
    for batch_dir in _all_batch_dirs(repo_root)[-max_batches:]:
        labels_path = os.path.join(batch_dir, "labels.csv")
        results_path = os.path.join(batch_dir, "results.json")
        records_by_key = {}
        if os.path.exists(results_path):
            with open(results_path) as f:
                for rec in json.load(f):
                    records_by_key[(rec.get("symbol"), rec.get("screener"))] = rec

        with open(labels_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("screener") != screener_name:
                    continue
                label = (row.get("human_label") or "").strip().lower()
                if label not in {"no", "n", "false", "0"}:
                    continue
                key = (row.get("symbol"), screener_name)
                false_positives.append({
                    "batch_date": row.get("batch_date"),
                    "symbol": row.get("symbol"),
                    "comment": row.get("comment") or "",
                    "record": records_by_key.get(key, row),
                })
    return false_positives


def load_missed_signals(screener_name, repo_root=REPO_ROOT, max_batches=10):
    """missed_signals/<date>/notes.csv rows whose screener_expected matches."""
    missed_root = os.path.join(repo_root, "missed_signals")
    rows = []
    for notes_path in sorted(glob.glob(os.path.join(missed_root, "*", "notes.csv")))[-max_batches:]:
        with open(notes_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("screener_expected") == screener_name:
                    rows.append(row)
    return rows


def render_rows_markdown(rows, fields=None):
    """Compact markdown-ish rendering of a list of dicts for prompt insertion."""
    if not rows:
        return "(none)"
    out = []
    for row in rows:
        if fields:
            shown = {k: row.get(k) for k in fields}
        else:
            shown = row
        out.append("- " + json.dumps(shown, default=str))
    return "\n".join(out)


def labeled_yes_signals(screener_name, repo_root=REPO_ROOT, max_batches=5):
    """Historical confirmed-good signals for this screener: (symbol, signal_date,
    batch_date) tuples where human_label == Yes, most recent batches first. Used
    by validate_no_regression as the regression-check set — a patch must not
    cause analyse() to stop firing on these.
    """
    out = []
    for batch_dir in reversed(_all_batch_dirs(repo_root)[-max_batches:]):
        labels_path = os.path.join(batch_dir, "labels.csv")
        with open(labels_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("screener") != screener_name:
                    continue
                label = (row.get("human_label") or "").strip().lower()
                if label in {"yes", "y", "true", "1"}:
                    out.append({
                        "symbol": row.get("symbol"),
                        "signal_date": row.get("signal_date"),
                        "batch_date": row.get("batch_date"),
                    })
    return out


# ---------------------------------------------------------------------------
# Regression validation
# ---------------------------------------------------------------------------

def _load_module_from_source(module_name, source, file_hint):
    """Write `source` to a temp file and import it fresh via importlib, without
    touching sys.modules for the real screener module (so the on-disk patch
    candidate can be evaluated before it's actually written into screeners/).
    """
    tmp_dir = tempfile.mkdtemp(prefix="patch_candidate_")
    tmp_path = os.path.join(tmp_dir, file_hint)
    with open(tmp_path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(module_name, tmp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_no_regression(screener_name, new_source, repo_root=REPO_ROOT, max_batches=5):
    """Re-run the PATCHED module (loaded from `new_source`, not from disk) against
    every historically-confirmed Yes signal for this screener, refetching fresh
    price history per symbol via yfinance. A regression is any previously-Yes
    symbol/date for which the patched analyse() now returns no signal at all.

    Returns {"ok": bool, "checked": int, "regressions": [...], "errors": [...]}.
    """
    import pandas as pd
    import yfinance as yf

    yes_signals = labeled_yes_signals(screener_name, repo_root, max_batches=max_batches)
    if not yes_signals:
        return {"ok": True, "checked": 0, "regressions": [], "errors": [],
                "note": "no labeled Yes signals to regression-check against yet"}

    try:
        mod = _load_module_from_source(
            f"_patch_candidate_{screener_name}", new_source, SCREENER_FILES[screener_name]
        )
    except Exception as e:
        return {"ok": False, "checked": 0, "regressions": [],
                "errors": [f"patched module failed to import: {e}"]}

    history_days = getattr(mod, "HISTORY_DAYS", 800)
    regressions, errors = [], []
    checked = 0

    for sig in yes_signals:
        symbol, signal_date = sig["symbol"], sig["signal_date"]
        if not symbol or not signal_date:
            continue
        as_of = pd.Timestamp(signal_date)
        start = (as_of - pd.Timedelta(days=history_days)).date()
        end = (as_of + pd.Timedelta(days=1)).date()
        try:
            df = yf.download(symbol + ".NS", start=str(start), end=str(end), interval="1d",
                              auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if df.empty:
                errors.append(f"{symbol}: no price history returned, skipped")
                continue
            result = mod.analyse(df, as_of)
            checked += 1
            if result is None:
                regressions.append({
                    "symbol": symbol, "signal_date": signal_date,
                    "batch_date": sig["batch_date"],
                    "reason": "patched analyse() no longer fires on a confirmed-Yes signal",
                })
        except Exception as e:
            errors.append(f"{symbol} ({signal_date}): {e}")

    return {"ok": len(regressions) == 0, "checked": checked,
            "regressions": regressions, "errors": errors}


# ---------------------------------------------------------------------------
# Git / PR plumbing (subprocess -> git + gh CLI; both preinstalled on
# GitHub-hosted runners, gh is pre-authed there via GH_TOKEN/GITHUB_TOKEN)
# ---------------------------------------------------------------------------

def _run(cmd, cwd=REPO_ROOT, check=True):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout.strip()


def create_branch(branch_name, repo_root=REPO_ROOT):
    _run(["git", "fetch", "origin", "main"], cwd=repo_root)
    _run(["git", "checkout", "-B", branch_name, "origin/main"], cwd=repo_root)
    return branch_name


def commit_and_push(branch_name, message, paths, repo_root=REPO_ROOT):
    _run(["git", "add", *paths], cwd=repo_root)
    _run(["git", "-c", "user.name=shelf-screener-debug-agent",
          "-c", "user.email=actions@users.noreply.github.com",
          "commit", "-m", message], cwd=repo_root)
    _run(["git", "push", "-u", "origin", branch_name], cwd=repo_root)


def open_pr(branch_name, title, body, base="main", repo_root=REPO_ROOT):
    """Requires `gh` CLI authenticated (GH_TOKEN env var set by the workflow)."""
    out = _run(["gh", "pr", "create", "--base", base, "--head", branch_name,
                "--title", title, "--body", body], cwd=repo_root)
    # `gh pr create` prints the PR URL as the last line of stdout.
    lines = [l for l in out.splitlines() if l.strip()]
    return lines[-1] if lines else out
