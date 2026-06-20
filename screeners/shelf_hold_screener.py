"""
Shelf-hold & bounce screener  (no undercut required)
with 50DMA-proximity score and sector grouping.

Universe: Nifty Midcap 150 + Nifty Smallcap 250 + Nifty Microcap 250

Pattern (SHELF_HOLD):
  steep recent advance -> short, HIGH consolidation ("shelf") just under the
  highs -> price TESTS the shelf from above (low approaches within APPROACH_PCT)
  without closing below it -> bounces back up.

How it differs from shelf_reclaim_screener.py:
  - NO undercut required: the shelf low is merely approached, not broken
  - A "test" day has: low within APPROACH_PCT (5%) of the shelf AND close above shelf
  - Tiny intraday wicks below the shelf (≤ MAX_WICK_BELOW) are allowed as long
    as the session CLOSES above the shelf
  - Catches higher-quality holds where the shelf never actually cracks; pairs
    with shelf_reclaim_screener (together they cover both hold and breach setups)
  - Same adaptive shelf detection as shelf_reclaim_screener (_find_shelf)
  - Same trend / prior-gain / turnover / volatility gates

Score (0-100, higher = better hold):
  30%  proximity     (how close the low got to the shelf; touching = full marks)
  20%  dry volume    (low test volume = distribution absent; 0.5× avg50 = max)
  20%  bounce day    (close well above shelf on the test day = conviction)
  15%  50DMA support (shelf close to rising 50DMA = extra conviction)
  15%  shelf quality (more pre-test touches = stronger level)
  +5   sector cluster (≥2 signals in the same industry today)

Usage:
  pip install yfinance pandas requests
  python shelf_hold_screener.py                # as of today
  python shelf_hold_screener.py 2026-05-25     # as of a past date
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
AS_OF_DATE        = None    # e.g. "2026-05-25"; None = today. CLI arg overrides.
TEST_WINDOW       = 10      # sessions to look for a shelf test
SHELF_LOOKBACK    = 30      # sessions before the test window searched for the shelf
MAX_SHELF_BELOW_PIVOT = 0.20  # shelf must sit within 20% of the 60d high
MIN_SHELF_TOUCHES = 2       # shelf tested >= 2 times before the test window
TOUCH_TOL         = 0.02    # tolerance for counting a shelf touch (±2%)
APPROACH_PCT      = 0.05    # low must come within 5% of the shelf to count as a test
MAX_WICK_BELOW    = 0.005   # allow tiny intraday wick ≤ 0.5% below shelf; close must be above
RECENT_DAYS       = 3       # only flag if the test happened in the last N sessions
BASE_WINDOW       = 60
MIN_PRIOR_GAIN    = 0.25
TREND_SLACK       = 0.97
MIN_AVG_TURNOVER  = 1e7
HISTORY_DAYS      = 800

# volatility cap (same as shelf_reclaim - accepts more volatile names)
MAX_ATR_PCT       = 9.0
SPIKY_RET_PCT     = 4.0

CONSTITUENT_URLS = [
    "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "https://niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
    "https://niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv",
]
# -----------------------------------------

def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__)) or "."
    except NameError:
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


def _count_visits(in_band_mask):
    """Count distinct visits to a level: consecutive bars in the band = 1 visit.

    E.g. [F,T,T,T,F,F,T,F] → 2 visits, not 4 bar-touches.
    This prevents a 4-day sideways cluster from inflating the touch count.
    """
    visits, inside = 0, False
    for b in in_band_mask:
        if b and not inside:
            visits += 1
            inside = True
        elif not b:
            inside = False
    return visits


def _find_shelf(pre_lows, pivot):
    """Adaptive shelf detection: (shelf, visits) or None.

    A candidate shelf is a low in `pre_lows` (the SHELF_LOOKBACK sessions
    before the test window) that:
      a) sits within MAX_SHELF_BELOW_PIVOT of the pivot ("high" shelf),
      b) was visited >= MIN_SHELF_TOUCHES distinct times (each contiguous run
         of lows within TOUCH_TOL counts as ONE visit, so a 4-day cluster
         on the level is 1 visit, not 4),
      c) HELD as support from its first visit to the end of the pre window
         (no intervening low breaks it by MAX_WICK_BELOW).
    Among valid candidates the most recently visited wins (tie: higher level).
    """
    best = None
    vals = pre_lows.values
    for level in vals:
        if level < pivot * (1 - MAX_SHELF_BELOW_PIVOT):
            continue
        in_band = (vals >= level) & (vals <= level * (1 + TOUCH_TOL))
        visits = _count_visits(in_band)
        if visits < MIN_SHELF_TOUCHES:
            continue
        first = int(in_band.argmax())
        # shelf must have held (no break during the lookback window)
        if (vals[first:] < level * (1 - MAX_WICK_BELOW)).any():
            continue
        last_touch = len(vals) - 1 - int(in_band[::-1].argmax())
        if best is None or (last_touch, level) > (best[0], best[1]):
            best = (last_touch, level, visits)
    return (best[1], best[2]) if best else None


def analyse(df, as_of):
    """SHELF_HOLD signal dict as of `as_of`, or None.

    Trend context is gated on the 50 DMA only (no 200 DMA dependency), so
    this applies uniformly regardless of listing age / history length.
    """
    df = df.dropna().loc[:as_of]
    if len(df) < 80:          # need enough history for MA50 + shelf detection
        return None

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    ma50 = c.rolling(50).mean()
    vol50 = v.rolling(50).mean()
    last  = c.iloc[-1]

    if (c * v).rolling(50).mean().iloc[-1] < MIN_AVG_TURNOVER:
        return None

    atr_pct = _atr_pct(h, l, c)
    if pd.isna(atr_pct) or atr_pct > MAX_ATR_PCT:
        return None
    rets = c.pct_change().iloc[-BASE_WINDOW:] * 100
    spiky_days = int((rets.abs() > SPIKY_RET_PCT).sum())

    # trend context: 50 DMA rising and price near/above it (TREND_SLACK allows
    # price to be testing the 50 DMA from slightly below — that IS the setup)
    trend_ok = (last > ma50.iloc[-1] * TREND_SLACK
                and ma50.iloc[-1] > ma50.iloc[-11])
    if not trend_ok:
        return None

    # momentum into the structure
    pivot = h.iloc[-BASE_WINDOW:].max()
    lookback = min(126, len(c))   # adaptive for new listings
    prior_gain = pivot / c.iloc[-lookback:].min() - 1
    if prior_gain < MIN_PRIOR_GAIN:
        return None

    # ---- the shelf: most recent tested resting level before the test window ----
    shelf_lows = l.iloc[-(SHELF_LOOKBACK + TEST_WINDOW):-TEST_WINDOW]
    if len(shelf_lows) < SHELF_LOOKBACK // 3:
        return None
    found = _find_shelf(shelf_lows, pivot)
    if found is None:
        return None
    shelf, touches = found

    # ---- the test window: look for an approach of the shelf (no significant undercut) ----
    test_l = l.iloc[-TEST_WINDOW:]
    test_c = c.iloc[-TEST_WINDOW:]
    test_v = v.iloc[-TEST_WINDOW:]

    # Approach: low came within APPROACH_PCT of the shelf from above
    approach_mask = test_l <= shelf * (1 + APPROACH_PCT)
    # Clean hold: low didn't break shelf by more than MAX_WICK_BELOW
    hold_mask     = test_l >= shelf * (1 - MAX_WICK_BELOW)
    # Close held above shelf on that day
    close_mask    = test_c > shelf

    valid_mask = approach_mask & hold_mask & close_mask
    if not valid_mask.any():
        return None

    # most recent valid test day
    t_pos = test_l[valid_mask].index[-1]
    t_i   = df.index.get_loc(t_pos)

    # must be recent
    sessions_since_test = len(df) - 1 - t_i
    if sessions_since_test > RECENT_DAYS:
        return None

    # current price still above shelf and not far below a rising MA50
    ma50_ok = last > ma50.iloc[-1] * TREND_SLACK
    if not (last > shelf and ma50_ok):
        return None

    # 50DMA proximity at test day
    ma50_at_test = ma50.iloc[t_i]
    ma_gap = (shelf / ma50_at_test - 1) if not pd.isna(ma50_at_test) else None

    # proximity of the low to the shelf on test day (0 = touched exactly, positive = above)
    low_at_test   = l.loc[t_pos]
    prox_pct      = (low_at_test / shelf - 1) * 100   # how far above shelf the low was
    close_at_test = c.loc[t_pos]
    bounce_pct    = (close_at_test / shelf - 1) * 100  # close vs shelf on test day
    test_vol_x    = v.loc[t_pos] / vol50.loc[t_pos] if not pd.isna(vol50.loc[t_pos]) else None

    return dict(
        signal="SHELF_HOLD",
        date=str(df.index[-1].date()),
        close=round(float(last), 2),
        atr_pct=round(float(atr_pct), 2),
        spiky_days=spiky_days,
        shelf=round(float(shelf), 2),
        shelf_touches=touches,
        test_date=str(pd.Timestamp(t_pos).date()),
        sessions_since_test=sessions_since_test,
        low_at_test=round(float(low_at_test), 2),
        prox_pct=round(float(prox_pct), 2),       # how close low got to shelf (lower = better)
        bounce_pct=round(float(bounce_pct), 2),    # close above shelf on test day
        test_vol_x=round(float(test_vol_x), 2) if test_vol_x is not None else None,
        ma50_gap_pct=round(float(ma_gap) * 100, 1) if ma_gap is not None else None,
        pct_from_pivot=round((last / pivot - 1) * 100, 1),
    )


def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def score_signals(out):
    """Add a 0-100 hold-quality score + sector cluster bonus."""

    def base_score(r):
        # Proximity: low touching shelf = 100, 3% above = 0
        prox_s  = _clip((APPROACH_PCT * 100 - r.prox_pct) / (APPROACH_PCT * 100) * 100)  # 0% above shelf=100, 5%=0
        # Dry volume: 0.5× avg50 = 100, 2× = 0
        vol_s   = _clip((2.0 - (r.test_vol_x or 1.0)) / 1.5 * 100)
        # Bounce: close 3%+ above shelf on test day = 100, 0% = 0
        bnc_s   = _clip(r.bounce_pct / 3.0 * 100)
        # 50DMA: shelf within 2% of ma50 = 100, 10%+ above = 0
        gap     = abs(r.ma50_gap_pct or 10.0)
        ma_s    = _clip((10.0 - gap) / 8.0 * 100)
        # Shelf quality: 2 touches = 0, 5+ touches = 100
        shelf_s = _clip((r.shelf_touches - 2) / 3 * 100)
        return (0.30 * prox_s + 0.20 * vol_s + 0.20 * bnc_s
                + 0.15 * ma_s  + 0.15 * shelf_s)

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
        end   = (as_of + pd.Timedelta(days=1)).date()
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
        tag = "  [SECTOR CCLUSTER]" if grp["sector_cluster"].iloc[0] and len(grp) >= 2 else ""
        print(f"\n--- {ind}  ({len(grp)} signal{'s' if len(grp) > 1 else ''}){tag}")
        print(grp[cols].to_string(index=False))


def main():
    as_of_str = AS_OF_DATE
    if len(sys.argv) > 1 and sys.argv[1] != "-f":
        try:
            pd.Timestamp(sys.argv[1])
            as_of_str = sys.argv[1]
        except ValueError:
            print(f"Warning: ignoring invalid argument '{sys.argv[1]}'")
    as_of = pd.Timestamp(as_of_str) if as_of_str else pd.Timestamp.today().normalize()

    out = run_scan(as_of)
    if out.empty:
        print(f"\nNo shelf holds as of {as_of.date()}.")
        return

    out = out.sort_values("score", ascending=False)
    fname = f"shelf_hold_results_{as_of.date()}.csv"
    out.to_csv(os.path.join(_script_dir(), fname), index=False)

    pd.set_option("display.width", 240)
    print(f"\n{len(out)} shelf holds as of {as_of.date()}  (saved to {fname})")
    print_grouped(out)
    print("\nscore = hold quality (proximity 30% + dry volume 20% + bounce 20% + "
          "50DMA support 15% + shelf tests 15% + sector-cluster bonus). "
          "prox_pct = how far above the shelf the test-day low was (lower = closer touch). "
          "bounce_pct = close above shelf on the test day. "
          "test_vol_x = test-day volume vs 50-session average (< 1 = dry = good). "
          "atr_pct = 14d ATR as % of price.")


if __name__ == "__main__":
    main()
