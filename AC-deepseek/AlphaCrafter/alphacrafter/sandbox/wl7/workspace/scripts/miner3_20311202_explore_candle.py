"""miner_3 exploratory cycle at 2031-12-02: OHLC candle-location / range factors.
Uses open/high/low/close; low corr with momentum library. Validated on 15-asset
cross-asset universe with same admission gates.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (ASSETS, DATA_DIR, load_close, load_macro, forward_ret,
                          daily_ic, ic_stats, rank_turnover, coverage_stats,
                          library_panel, max_lib_corr, master_calendar)

END = "2031-12-01"
close = load_close(END)
macro = load_macro(END)
lib = library_panel(close, macro)
cal = master_calendar(END)

def load_field(field):
    out = pd.DataFrame(index=cal)
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")[field].reindex(cal).ffill()
        out[a] = s
    return out

opn = load_field("open"); hi = load_field("high")
lo = load_field("low"); vol = load_field("volume")
ret = close.pct_change()
rng = (hi - lo).replace(0, np.nan)

def close_loc(window=20, skip=5):
    """Mean intraday close-location (close-low)/(high-low) shipped skip, demeaned."""
    cl = (close - lo) / rng
    s = cl.shift(skip).rolling(window, min_periods=10).mean()
    return s.subtract(s.median(axis=1), axis=0)

def range_expansion(window=20):
    """Current range vs 60d median range ratio, demeaned (range breakout)."""
    rmed = rng.rolling(60, min_periods=20).median()
    rr = rng / rmed
    s = rr.rolling(window, min_periods=5).mean()
    return s.subtract(s.median(axis=1), axis=0)

def lower_shadow(window=20, skip=5):
    """Mean lower-shadow / range (buyer defense at lows), shipped skip, demeaned."""
    lower = (close - lo) / rng
    s = lower.shift(skip).rolling(window, min_periods=10).mean()
    return s.subtract(s.median(axis=1), axis=0)

def up_gap(window=20):
    """Mean positive open-gap over window, demeaned."""
    gap = open_gap = opn / close.shift(1) - 1.0
    s = np.where(gap > 0, gap, 0.0)
    s = pd.DataFrame(s, index=gap.index, columns=gap.columns)
    s = s.rolling(window, min_periods=10).mean()
    return s.subtract(s.median(axis=1), axis=0)

cands = {"close_loc_20x5": close_loc, "range_expansion_20": range_expansion,
         "lower_shadow_20x5": lower_shadow, "up_gap_20": up_gap}

print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")
fwd = forward_ret(close, 10)

for name, fn in cands.items():
    f = fn()
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10)
    cov = coverage_stats(f, fwd); turn = rank_turnover(f, 10)
    best, pairs = max_lib_corr(f, lib)
    f_5 = f.tail(500); s5 = ic_stats(daily_ic(f_5, forward_ret(close,10).reindex(f_5.index)), 10)
    f_2 = f.tail(250); s2 = ic_stats(daily_ic(f_2, forward_ret(close,10).reindex(f_2.index)), 10)
    gate = abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840
    print("\n=== %s ===" % name)
    print(f"IC10={st['ic']:+.4f} ICIR10={st['icir']:+.3f} hit={st['hit']:.2f} n={st['n']} "
          f"covAD={cov['coverage_asset_days']:.2f} covD8={cov['coverage_dates_ge8']:.2f} turn={turn:.2f}")
    print(f"recent500 IC={s5['ic']:+.4f}/{s5['icir']:+.3f}  recent250 IC={s2['ic']:+.4f}/{s2['icir']:+.3f}")
    print(f"max_abs_lib_corr={best:.3f}  GATE={'PASS' if gate else 'FAIL'}")
    print("corr_pairs:", {k: round(v,3) for k,v in sorted(pairs.items(), key=lambda x:-abs(x[1]))[:4]})