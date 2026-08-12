"""miner_1 screening: multiple candidate factor constructions (exploration only).
Not a persistence decision. Focus: trend persistence, drawdown reversal,
risk-adjusted momentum, trend-consistent momentum.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner1_common import load_closes, load_macro, forward_returns, daily_ic, summarize_ic

END = "2029-01-18"
px = load_closes(end_date=END)
mx = load_macro(end_date=END)
rets = px.pct_change()
fwd = forward_returns(px, horizons=(1, 5, 10))

def kaufman_er(px, n=60):
    """Efficiency ratio: |close_t - close_{t-n}| / sum(|dclose|) over n."""
    move = (px - px.shift(n)).abs()
    path = px.diff().abs().rolling(n).sum()
    return move / path.replace(0, np.nan)

def max_drawdown(px, n=90):
    """Negative max drawdown over trailing n days (>=0 drawdown = no loss)."""
    roll_max = px.rolling(n, min_periods=20).max()
    dd = px / roll_max - 1.0
    return dd.rolling(n, min_periods=20).min()

def sharpe_mom(px, n=60):
    r = px.pct_change()
    mu = (px / px.shift(n) - 1.0)
    vol = r.rolling(n).std() * np.sqrt(252)
    return mu / vol.replace(0, np.nan)

def trend_consistent_mom(px, n=120, skip=5, short=20):
    """120d momentum (skip 5) scaled by sign agreement with 20d momentum."""
    mom = px.shift(skip) / px.shift(n + skip) - 1.0
    mom_s = px / px.shift(short) - 1.0
    return mom * np.sign(mom_s)

def vol_scaled_rev(px, n=5):
    """5d reversal scaled by realized vol (contrarian)."""
    r = px.pct_change()
    rev = px.shift(n) / px - 1.0  # negative recent return -> high value
    vol = r.rolling(20).std()
    return rev / vol.replace(0, np.nan)

def vix_cond_mom(px, mx, n=120, skip=5):
    """Momentum conditioned by VIX regime (high VIX -> reversal tilt)."""
    mom = px.shift(skip) / px.shift(n + skip) - 1.0
    vix = mx["VIX"].reindex(px.index).ffill()
    vix_z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
    w = np.where(vix_z < 0, 1.0, -0.5)  # low VIX: momentum; high VIX: fade
    return mom * pd.Series(w, index=px.index)

candidates = {
    "er_60": kaufman_er(px, 60),
    "dd_90": max_drawdown(px, 90),
    "dd_60": max_drawdown(px, 60),
    "sharpe_mom_60": sharpe_mom(px, 60),
    "trend_mom_120s5": trend_consistent_mom(px, 120, 5, 20),
    "vol_rev_5": vol_scaled_rev(px, 5),
    "vix_cond_mom": vix_cond_mom(px, mx),
}

for name, f in candidates.items():
    print("=" * 70)
    print("CANDIDATE:", name)
    res = {}
    for h in (1, 5, 10):
        ic = daily_ic(f, fwd[f"fwd{h}"])
        res[f"h{h}"] = summarize_ic(ic, name)
    ic1 = res["h1"]
    print(f"  IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit1={ic1['hit']:.3f} n={ic1['n_dates']}")
    print(f"  IC5={res['h5']['ic']:+.4f} ICIR5={res['h5']['icir']:+.3f} | IC10={res['h10']['ic']:+.4f} ICIR10={res['h10']['icir']:+.3f}")
    cov_m, cov_min = f.notna().mean(axis=1).mean(), f.notna().mean(axis=1).min()
    print(f"  coverage mean={cov_m:.3f} min={cov_min:.3f}")
