"""
Shelf-undercut & reclaim screener (V-flush shakeouts in fast movers)
with 50DMA-tag requirement, reclaim-quality score, and sector grouping.

Universe: Nifty Midcap 150 + Nifty Smallcap 250 + Nifty Microcap 250

Pattern (SHELF_RECLAIM):
  steep recent advance -> short, HIGH consolidation ("shelf") just under the
  highs -> violent flush undercuts the shelf low and tags the rising 50DMA
  -> price reclaims the shelf within a few sessions.

How it differs from shakeout_screener.py:
  - the floor is the most recent resting level (shelf), not the 60-day low;
    catches shakeouts in stocks that ran hard recently, where the 60-day low
    sits far below the actual traded support
  - the shelf is detected adaptively: the most recent tested level (>= 2 low
    touches within tolerance) that HELD as support until the flush - not the
    min of a fixed window, which in fast movers lands on the ramp low
  - the 50DMA tag is scored, not a hard gate (closer tag = higher score);
    only flush lows still > 10% above the 50DMA are rejected outright
  - looser volatility cap (ATR <= 9%): violent flushes are the target here,
    so this screener deliberately accepts what the "clean mover" filter rejects

Score (0-100, higher = higher-quality reclaim):
  25%  reclaim speed    (same/next-day reclaim of the shelf = full marks)
  20%  undercut wick    (close position in the undercut candle's range)
  15%  50DMA tag        (closer tag of the MA at the low = better)
  15%  undercut volume  (bigger flush = more stops cleared)
  15%  reclaim volume   (demand on the recovery)
  10%  shelf quality    (more tests of the shelf before the flush = better)
  +5   sector cluster   (>=2 signals from the same industry today)

Usage:
  pip install yfinance pandas requests
  python shelf_reclaim_screener.py                # as of today
  python shelf_reclaim_screener.py 2026-05-13     # as of a past date
  In Colab: set AS_OF_DATE below and run.
"""

import io
import os
import sys
import datetime as dt

import pandas as pd
import requests
import yfinance as yf

# ---------------- config ----------------
AS_OF_DATE        = None    # e.g. "2026-05-13"; None = today. CLI arg overrides.
FLUSH_WINDOW      = 10      # sessions to look for the undercut (the flush)
SHELF_LOOKBACK    = 30      # sessions before the flush window searched for the shelf
MAX_SHELF_BELOW_PIVOT = 0.20  # shelf low must sit within 20% of the 60d high ("high" shelf)
MIN_SHELF_TOUCHES = 2       # shelf must be tested >= 2 times (lows within TOUCH_TOL)
TOUCH_TOL         = 0.02
MIN_UNDERCUT      = 0.005   # flush low must break the shelf by >= 0.5%
MAX_UNDERCUT      = 0.10    # ...but not by more than 10%
RECLAIM_DAYS      = 5       # close back above the shelf within N sessions of the undercut
RECENT_DAYS       = 3       # only flag if the reclaim happened in the last N sessions
MA_TAG_PCT        = 0.02    # 50DMA-tag distance that earns the full tag score
MA_TAG_HARD       = 0.10    # reject only if the flush low is > 10% above the 50DMA
BASE_WINDOW       = 60
MIN_PRIOR_GAIN    = 0.25
TREND_SLACK       = 0.97
MIN_AVG_TURNOVER  = 1e7
HISTORY_DAYS      = 800

# volatility cap - deliberately looser than the other screeners
MAX_ATR_PCT       = 9.0    # reject only the wildest names (14d ATR > 9% of price)
SPIKY_RET_PCT     = 4.0    # |daily return| > 4% counts as a "spiky" day (info)

CONSTITUENT_URLS = [
    "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "https://niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
    "https://niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv",
]
# -----------------------------------------

def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__)) or "."
    except NameError:                      # Colab / Jupyter
        return "."


def get_universe():
    """Return {symbol: industry} from niftyindices.com (falls back to local CSVs)."""
    universe = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in CONSTITUENT_URLS:
        local = os.path.join(_script_dir(), url.rsplit("/", 1)[-1])
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            if os.path.exists(local):
                print(f"  download failed ({e}); using local {local}")
                df = pd.read_csv(local)
            else:
                print(f"  WARNING: could not load {url} ({e})")
                continue
        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        ind_col = "Industry" if "Industry" in df.columns else None
        for _, row in df.iterrows():
            sym = str(row[sym_col]).strip()
            universe[sym] = (str(row[ind_col]).strip()
                             if ind_col and pd.notna(row[ind_col]) else "Unknown")
    return universe


def _atr_pct(h, l, c, n=14):
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return (tr.rolling(n).mean() / c * 100).iloc[-1]


def _find_shelf(pre_lows, pivot):
    """Adaptive shelf detection: (shelf, touches) or None.

    A candidate shelf is a low in `pre_lows` (the SHELF_LOOKBACK sessions
    before the flush window) that:
      a) sits within MAX_SHELF_BELOW_PIVOT of the pivot ("high" shelf),
      b) was tested >= MIN_SHELF_TOUCHES times (lows within TOUCH_TOL above),
      c) HELD as support from its first touch to the end of the pre window
         (no intervening low breaks it by MIN_UNDERCUT).
    Among valid candidates the most recently touched wins (tie: higher level).
    This replaces min-of-fixed-window, which in fast movers returns the ramp
    low far below the actual traded support.
    """
    best = None        # (last_touch_idx, level, touches)
    vals = pre_lows.values
    for level in vals:
        if level < pivot * (1 - MAX_SHELF_BELOW_PIVOT):
            continue
        in_band = (vals >= level) & (vals <= level * (1 + TOUCH_TOL))
        touches = int(in_band.sum())
        if touches < MIN_SHELF_TOUCHES:
            continue
        first = int(in_band.argmax())
        if (vals[first:] < level * (1 - MIN_UNDERCUT)).any():
            continue                    # broke before the flush - not a shelf
        last_touch = len(vals) - 1 - int(in_band[::-1].argmax())
        if best is None or (last_touch, level) > (best[0], best[1]):
            best = (last_touch, level, touches)
    return (best[1], best[2]) if best else None


def analyse(df, as_of):
    """SHELF_RECLAIM signal dict as of `as_of`, or None."""
    df = df.dropna().loc[:as_of]
    if len(df) < 60:
        return None

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    ma50 = c.rolling(50).mean()
    vol50 = v.rolling(50).mean()
    last = c.iloc[-1]

    if (c * v).rolling(50).mean().iloc[-1] < MIN_AVG_TURNOVER:
        return None

    # volatility cap (loose by design)
    atr_pct = _atr_pct(h, l, c)
    if pd.isna(atr_pct) or atr_pct > MAX_ATR_PCT:
        return None
    rets = c.pct_change().iloc[-BASE_WINDOW:] * 100
    spiky_days = int((rets.abs() > SPIKY_RET_PCT).sum())

    # trend context: rising 50DMA
    if not (ma50.iloc[-1] > ma50.iloc[-11]):
        return None

    # momentum into the structure
    pivot = h.iloc[-BASE_WINDOW:].max()
    prior_gain = pivot / c.iloc[-126:].min() - 1
    if prior_gain < MIN_PRIOR_GAIN:
        return None

    # ---- the shelf: most recent tested resting level before the flush ----
    shelf_lows = l.iloc[-(SHELF_LOOKBACK + FLUSH_WINDOW):-FLUSH_WINDOW]
    if len(shelf_lows) < SHELF_LOOKBACK // 3:
        return None
    found = _find_shelf(shelf_lows, pivot)
    if found is None:
        return None
    shelf, touches = found

    # ---- the flush: undercut of the shelf inside the recent window ----
    flush_l = l.iloc[-FLUSH_WINDOW:]
    under_mask = flush_l < shelf * (1 - MIN_UNDERCUT)
    if not under_mask.any():
        return None
    if 1 - flush_l.min() / shelf > MAX_UNDERCUT:        # deepest break too deep
        return None
    u_pos = flush_l[under_mask].index[-1]               # most recent undercut day
    u_i = df.index.get_loc(u_pos)
    depth = 1 - l.loc[u_pos] / shelf

    # 50DMA tag: scored (closer = better), hard-reject only far-above flushes.
    # A steep advance leaves the 50DMA lagging well below price, so a valid
    # flush can bottom several % above it - don't hard-gate at 2%.
    ma_gap = l.loc[u_pos] / ma50.iloc[u_i] - 1
    if ma_gap > MA_TAG_HARD:
        return None

    # ---- the reclaim: close back above the shelf, recently ----
    after = c.iloc[u_i: u_i + 1 + RECLAIM_DAYS]
    reclaim_mask = after > shelf
    if not reclaim_mask.any():
        return None
    reclaim_pos = after.index[reclaim_mask.values.argmax()]
    r_i = df.index.get_loc(reclaim_pos)
    if len(df) - 1 - r_i > RECENT_DAYS:
        return None
    if not (last > shelf and last > ma50.iloc[-1]):
        return None

    rng = h.loc[u_pos] - l.loc[u_pos]
    wick_pos = (c.loc[u_pos] - l.loc[u_pos]) / rng if rng > 0 else 0.0

    return dict(signal="SHELF_RECLAIM",
                date=str(df.index[-1].date()),
                close=round(last, 2),
                atr_pct=round(float(atr_pct), 2),
                spiky_days=spiky_days,
                shelf=round(shelf, 2),
                shelf_touches=touches,
                undercut_date=str(pd.Timestamp(u_pos).date()),
                undercut_pct=round(depth * 100, 1),
                ma50_gap_pct=round(ma_gap * 100, 1),
                days_to_reclaim=int(r_i - u_i),
                wick_close_pos=round(float(wick_pos), 2),
                undercut_vol_x=round(v.loc[u_pos] / vol50.loc[u_pos], 1),
                reclaim_vol_x=round(v.loc[reclaim_pos] / vol50.loc[reclaim_pos], 1),
                pct_from_pivot=round((last / pivot - 1) * 100, 1))


def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def score_signals(out):
    """Add a 0-100 reclaim-quality score + sector cluster bonus."""
    reclaim_map = {0: 100, 1: 90, 2: 75, 3: 50, 4: 30, 5: 15}

    def base_score(r):
        rec_s = reclaim_map.get(int(r.days_to_reclaim), 10)
        wick_s = _clip(r.wick_close_pos * 100)
        # full marks within MA_TAG_PCT of the 50DMA, linear to 0 at MA_TAG_HARD
        gap = max(abs(r.ma50_gap_pct), MA_TAG_PCT * 100)
        tag_s = _clip((MA_TAG_HARD * 100 - gap)
                      / ((MA_TAG_HARD - MA_TAG_PCT) * 100) * 100)
        uvol_s = _clip(r.undercut_vol_x / 3 * 100)
        rvol_s = _clip(r.reclaim_vol_x / 2 * 100)
        shelf_s = _clip((r.shelf_touches - 1) / 2 * 100)
        return (0.25 * rec_s + 0.20 * wick_s + 0.15 * tag_s
                + 0.15 * uvol_s + 0.15 * rvol_s + 0.10 * shelf_s)

    out["score"] = out.apply(base_score, axis=1)
    counts = out.groupby("industry")["symbol"].transform("count")
    out["sector_cluster"] = counts >= 2
    out["score"] = (out["score"] + out["sector_cluster"] * 5).clip(upper=100).round(1)
    return out


def run_scan(as_of=None, data=None, universe=None):
    """Scan as of a date. Pass data/universe to reuse downloads when looping."""
    as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()

    if universe is None:
        print("Loading universe (Midcap 150 + Smallcap 250 + Microcap 250)...")
        universe = get_universe()
        if not universe:
            sys.exit("No symbols loaded - aborting.")
        print(f"  {len(universe)} symbols")
    symbols = sorted(universe)

    if data is None:
        start = (as_of - pd.Timedelta(days=HISTORY_DAYS)).date()
        end = (as_of + pd.Timedelta(days=1)).date()
        print(f"Downloading daily data {start} -> {as_of.date()} from Yahoo...")
        data = yf.download([s + ".NS" for s in symbols], start=str(start),
                           end=str(end), interval="1d", group_by="ticker",
                           auto_adjust=True, threads=True, progress=True)

    results = []
    for s in symbols:
        try:
            df = data[s + ".NS"] if isinstance(data.columns, pd.MultiIndex) else data
            sig = analyse(df, as_of)
            if sig:
                results.append({"symbol": s, "industry": universe.get(s, "Unknown"), **sig})
        except Exception:
            continue

    out = pd.DataFrame(results)
    if not out.empty:
        out = score_signals(out)
    return out


def print_grouped(out):
    """Print signals clubbed by industry, best industries first."""
    order = (out.groupby("industry")["score"].max()
             .sort_values(ascending=False).index)
    cols = [c for c in out.columns if c not in ("industry", "sector_cluster")]
    for ind in order:
        grp = out[out["industry"] == ind].sort_values("score", ascending=False)
        tag = "  [SECTOR CLUSTER]" if grp["sector_cluster"].iloc[0] and len(grp) >= 2 else ""
        print(f"\n--- {ind}  ({len(grp)} signal{'s' if len(grp) > 1 else ''}){tag}")
        print(grp[cols].to_string(index=False))


def main():
    as_of_str = AS_OF_DATE
    if len(sys.argv) > 1 and sys.argv[1] != "-f":      # ignore Jupyter kernel args
        try:
            pd.Timestamp(sys.argv[1])
            as_of_str = sys.argv[1]
        except ValueError:
            print(f"Warning: ignoring invalid argument '{sys.argv[1]}'")
    as_of = pd.Timestamp(as_of_str) if as_of_str else pd.Timestamp.today().normalize()

    out = run_scan(as_of)
    if out.empty:
        print(f"\nNo shelf reclaims as of {as_of.date()}.")
        return

    out = out.sort_values("score", ascending=False)
    fname = f"shelf_reclaim_results_{as_of.date()}.csv"
    out.to_csv(os.path.join(_script_dir(), fname), index=False)

    pd.set_option("display.width", 220)
    print(f"\n{len(out)} shelf reclaims as of {as_of.date()}  (saved to {fname})")
    print_grouped(out)
    print("\nscore = reclaim quality (reclaim speed 25% + undercut wick 20% + "
          "50DMA tag 15% + flush volume 15% + reclaim volume 15% + shelf tests 10% + "
          "sector-cluster bonus). 50DMA tag: full marks within 2% of the MA, "
          "fading to 0 at 10% (flushes > 10% above the 50DMA are rejected). "
          "atr_pct = 14d ATR as % of price (cap 9 - this "
          "screener accepts volatile flushes by design). "
          "spiky_days = days with >4% moves in the last 60 sessions.")


if __name__ == "__main__":
    main()
