"""miner_3 exploratory cycle at 2031-12-02 (visible through 2031-12-01).

Novel family: volume-participation / accumulation-pressure factors that use
volume data (not price-only) and are lowly correlated with the active momentum
library. We test several interpretable constructions cross-sectionally on the
15-asset tradable universe and gate them on |IC|>=0.0070, |ICIR|>=0.0840.
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

# load volume on master calendar
cal = master_calendar(END)
vol = pd.DataFrame(index=cal)
for a in ASSETS:
    df = pd.read_csv(f"{DATA_DIR}/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["volume"].reindex(cal).ffill()
    vol[a] = s

ret = close.pct_change()

def up_vol_ratio(window=20, skip=5):
    """Mean up-day volume fraction over rolling window (ship skip)."""
    r = ret.shift(skip)
    up = vol.where(r > 0, 0.0)
    tot = vol.rolling(window, min_periods=10).sum()
    upsum = up.rolling(window, min_periods=10).sum()
    return upsum / tot

def up_vol_ratio_x_mom(window=20, skip=5):
    """Up-volume pressure interacted with relative momentum: only up-press score
    on assets that already trend (downside-protected participation tilt)."""
    uvr = up_vol_ratio(window, skip)
    mom = close / close.shift(window + skip) - 1.0
    relmom = mom.subtract(mom.median(axis=1), axis=0)
    sign = np.sign(relmom)
    return uvr * sign

def vol_trend_quality(window=20):
    """Return-per-unit-volume trend: slope of cumulative (price scikit proxy).
    Implements a simple volume-weighted change: mean(pct_change*log1p(vol/med))."""
    r = ret
    v_ratio = vol / vol.rolling(60, min_periods=10).median()
    score = (r * np.log1p(v_ratio)).rolling(window, min_periods=10).sum()
    return score.subtract(score.median(axis=1), axis=0)

cands = {
    "up_vol_ratio_20x5": up_vol_ratio,
    "up_vol_x_mom_20x5": up_vol_ratio_x_mom,
    "vol_trend_quality_20": vol_trend_quality,
}

print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")
fwd = forward_ret(close, 10)

for name, fn in cands.items():
    f = fn()
    ic = daily_ic(f, fwd)
    st = ic_stats(ic, 10)
    cov = coverage_stats(f, fwd)
    turn = rank_turnover(f, 10)
    best, pairs = max_lib_corr(f, lib)
    # recent windows
    f_5 = f.tail(500); s5 = ic_stats(daily_ic(f_5, forward_ret(close,10).reindex(f_5.index)), 10)
    f_2 = f.tail(250); s2 = ic_stats(daily_ic(f_2, forward_ret(close,10).reindex(f_2.index)), 10)
    gate = abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840
    print("\n=== %s ===" % name)
    print(f"IC10={st['ic']:+.4f} ICIR10={st['icir']:+.3f} hit={st['hit']:.2f} n={st['n']} "
          f"covAD={cov['coverage_asset_days']:.2f} covD8={cov['coverage_dates_ge8']:.2f} turn={turn:.2f}")
    print(f"recent500 IC={s5['ic']:+.4f}/{s5['icir']:+.3f}  recent250 IC={s2['ic']:+.4f}/{s2['icir']:+.3f}")
    print(f"max_abs_lib_corr={best:.3f}  GATE={'PASS' if gate else 'FAIL'}")
    print("corr_pairs:", {k: round(v,3) for k,v in sorted(pairs.items(), key=lambda x:-abs(x[1]))[:4]})