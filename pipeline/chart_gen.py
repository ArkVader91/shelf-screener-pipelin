"""
chart_gen.py — stage 2 of the shelf-screener feedback loop.

Reads review_batches/<date>/results.json (written by run_batch.py) and renders
one annotated candlestick chart per signal:

  review_batches/<date>/charts/<SYMBOL>_<SCREENER>.png

Per the project's charting rules (CLAUDE.md):
  - candlesticks only, never a line chart
  - 20 / 50 / 200 DMA overlaid
  - enough history is fetched before the visible window that the 200 DMA is
    plotted across the FULL visible range, not just a partial tail

On top of that (design doc section 3.2), this draws the shelf level as a
horizontal line, marks the bars that counted as shelf touches, and marks the
test/undercut day and (for FLEX/STRICT) the reclaim day — the same context a
human reviewer needs to judge whether a signal "looks right" without having
to recompute anything by hand.

Implemented with hand-rolled matplotlib (no mplfinance dependency) so it has
no extra package requirement beyond pandas/matplotlib/yfinance.

Usage:
  python pipeline/chart_gen.py --date 2026-06-19
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from screeners import shelf_hold_screener as hold_mod          # noqa: E402
from screeners import shelf_reclaim_screener as flex_mod       # noqa: E402
from screeners import shelf_reclaim_screener2 as strict_mod    # noqa: E402

SCREENER_MODULES = {"HOLD": hold_mod, "FLEX": flex_mod, "STRICT": strict_mod}

VISIBLE_DAYS = 180   # trading days shown on the x-axis
HISTORY_DAYS = 800   # calendar days fetched -- per CLAUDE.md, enough that the
                     # 200 DMA covers the full visible window, not a partial tail

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"


def fetch_history(symbol, as_of):
    import yfinance as yf
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=HISTORY_DAYS)).date()
    end = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).date()
    df = yf.download(symbol + ".NS", start=str(start), end=str(end), interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def find_date_pos(df, date_str):
    """Positional index of date_str in df, or the nearest prior trading day."""
    if not date_str:
        return None
    ts = pd.Timestamp(date_str)
    if ts in df.index:
        return df.index.get_loc(ts)
    idx = df.index[df.index <= ts]
    if len(idx) == 0:
        return None
    return df.index.get_loc(idx[-1])


def draw_candles(ax, df):
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    x = np.arange(len(df))
    up = c >= o
    colors = np.where(up, UP_COLOR, DOWN_COLOR)
    ax.vlines(x, l, h, color=colors, linewidth=1, zorder=2)
    body_bottom = np.minimum(o, c)
    body_height = np.abs(c - o)
    min_sliver = (h.max() - l.min()) * 0.0015 if len(h) else 0.01
    body_height = np.where(body_height < min_sliver, min_sliver, body_height)
    ax.bar(x, body_height, bottom=body_bottom, width=0.6, color=colors, zorder=3, linewidth=0)
    return x


def annotate_shelf(ax, df, x, rec, mod, screener):
    """Shelf line, recomputed touch markers, and the test/undercut (+reclaim) day."""
    shelf = rec.get("shelf")
    if not shelf:
        return
    ax.axhline(shelf, color="#555555", linestyle="--", linewidth=1.1,
               label=f"Shelf {shelf:.2f}", zorder=1)

    lookback = getattr(mod, "SHELF_LOOKBACK", 30)
    gap_window = getattr(mod, "TEST_WINDOW", None) or getattr(mod, "FLUSH_WINDOW", 10)
    tol = getattr(mod, "TOUCH_TOL", 0.02)

    n = len(df)
    win_start = max(0, n - (lookback + gap_window))
    win_end = max(0, n - gap_window)
    lows_win = df["Low"].iloc[win_start:win_end]
    mask = (lows_win >= shelf) & (lows_win <= shelf * (1 + tol))
    touch_positions = [win_start + j for j, m in enumerate(mask.values) if m]
    if touch_positions:
        ax.scatter([x[i] for i in touch_positions],
                   [df["Low"].iloc[i] for i in touch_positions],
                   marker="^", s=70, color="#555555", zorder=5, label="Shelf touch")

    if screener == "HOLD":
        pos = find_date_pos(df, rec.get("test_date"))
        if pos is not None:
            ax.scatter([x[pos]], [df["Low"].iloc[pos]], marker="*", s=240,
                       color="#d81b60", zorder=6, label="Test day")
    else:
        pos = find_date_pos(df, rec.get("undercut_date"))
        if pos is not None:
            ax.scatter([x[pos]], [df["Low"].iloc[pos]], marker="*", s=240,
                       color="#d81b60", zorder=6, label="Undercut day")
            r_pos = pos + int(rec.get("days_to_reclaim") or 0)
            if 0 <= r_pos < n:
                ax.scatter([x[r_pos]], [df["Close"].iloc[r_pos]], marker="o", s=100,
                           facecolors="none", edgecolors="#2e7d32", linewidths=2,
                           zorder=6, label="Reclaim day")


def render_chart(symbol, screener, rec, out_path):
    mod = SCREENER_MODULES[screener]
    as_of = pd.Timestamp(rec.get("signal_date") or rec.get("date"))
    df_full = fetch_history(symbol, as_of)
    if df_full.empty or len(df_full) < 60:
        print(f"  skip {symbol}/{screener}: not enough history ({len(df_full)} bars)")
        return False
    df_full = df_full.loc[:as_of]

    ma20 = df_full["Close"].rolling(20).mean()
    ma50 = df_full["Close"].rolling(50).mean()
    ma200 = df_full["Close"].rolling(200).mean()

    n_full = len(df_full)
    vis_start = max(0, n_full - VISIBLE_DAYS)
    df = df_full.iloc[vis_start:]
    if len(df) < 5:
        print(f"  skip {symbol}/{screener}: visible window too short")
        return False

    fig, ax = plt.subplots(figsize=(14, 7), dpi=130)
    x = draw_candles(ax, df)
    ax.plot(x, ma20.iloc[vis_start:].values, color="#1e88e5", linewidth=1.1, label="20 DMA")
    ax.plot(x, ma50.iloc[vis_start:].values, color="#fb8c00", linewidth=1.3, label="50 DMA")
    ax.plot(x, ma200.iloc[vis_start:].values, color="#8e24aa", linewidth=1.5, label="200 DMA")

    annotate_shelf(ax, df, x, rec, mod, screener)

    tick_step = max(1, len(df) // 10)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in df.index[::tick_step]],
                       rotation=30, ha="right")
    ax.set_xlim(-1, len(df))
    ax.margins(y=0.08)

    score = rec.get("score")
    score_str = f" · score {score}" if score is not None else ""
    ax.set_title(f"{symbol}  ·  SHELF_{screener}  ·  signal {rec.get('date')}{score_str}",
                 fontsize=12)
    ax.set_ylabel("Price (Rs.)")
    ax.grid(alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None, help="Batch date (YYYY-MM-DD); default = today")
    ap.add_argument("--batch-root", default=os.path.join(REPO_ROOT, "review_batches"))
    args = ap.parse_args()

    batch_date = args.date or str(pd.Timestamp.today().normalize().date())
    batch_dir = os.path.join(args.batch_root, batch_date)
    results_path = os.path.join(batch_dir, "results.json")
    if not os.path.exists(results_path):
        sys.exit(f"No results.json for batch {batch_date} (looked in {results_path})")

    with open(results_path) as f:
        records = json.load(f)

    charts_dir = os.path.join(batch_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    ok = failed = 0
    for rec in records:
        symbol, screener = rec["symbol"], rec["screener"]
        out_path = os.path.join(charts_dir, f"{symbol}_{screener}.png")
        try:
            if render_chart(symbol, screener, rec, out_path):
                ok += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR charting {symbol}/{screener}: {e}")
            failed += 1

    print(f"\nCharts: {ok} written, {failed} failed/skipped -> {charts_dir}")


if __name__ == "__main__":
    main()
