"""miner_1 research cycle 2026-07-16: screen decorrelated risk/vol/trend factor families.

Library is dominated by 1-3d reversal variants + one 10d-skip5 momentum. Focus here is
families built on volatility structure, skew/risk asymmetry, drawdown and trend that are
likely decorrelated (pairwise rho < 0.5) from the reversal cluster.

Validation window: 2021-01-01 .. 2026-07-15 (warm-up ends), 15-name cross-asset panel.
Admission gates (daily): |IC| >= 0.0070, |ICIR| >= 0.0840.
"""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner1_common import (CUT, START, SYMBOLS, MACRO, DATA_DIR, IDX_DIR,
                           load_close, build_returns, factor_panel, ic_analysis,
                           decay_analysis, coverage, turnover, summary)

T0 = time.time()
closes = load_close()
print(f"loaded {len(closes)} symbols; data through {CUT.date()}")

# forward returns
fwd1 = build_returns(closes, 1)

# ---------------------------------------------------------------- factor fns
def rv(n):
    def f(df):
        r = df["close"].pct_change()
        return r.rolling(n).std() * np.sqrt(252)
    return f

def rv_ratio(short, long):
    def f(df):
        r = df["close"].pct_change()
        vs = r.rolling(short).std()
        vl = r.rolling(long).std()
        return vs / vl
    return f

def parkinson(n):
    def f(df):
        hl = np.log(df["high"] / df["low"])
        return hl.rolling(n).mean() * np.sqrt(252 * np.log(2)) if False else hl.rolling(n).std() * np.sqrt(252)
    return f

def semi_ratio(n):
    def f(df):
        r = df["close"].pct_change()
        down = r.where(r < 0).rolling(n).std()
        up = r.where(r > 0).rolling(n).std()
        return down / up
    return f

def skew(n):
    def f(df):
        return df["close"].pct_change().rolling(n).skew()
    return f

def downside_vol(n):
    def f(df):
        r = df["close"].pct_change()
        return r.where(r < 0).rolling(n).std() * np.sqrt(252)
    return f

def vol_ratio_to_panel(n):
    """Cross-sectional: asset realized vol relative to panel median realized vol."""
    def f(df):
        r = df["close"].pct_change()
        return r.rolling(n).std() * np.sqrt(252)
    return f

def drawdown(n):
    def f(df):
        return df["close"] / df["close"].rolling(n).max() - 1.0
    return f

def dd_speed(n):
    """How fast the drawdown is developing: 1d change in drawdown depth."""
    def f(df):
        dd = df["close"] / df["close"].rolling(n).max() - 1.0
        return dd.diff()
    return f

def ma_dist(n):
    def f(df):
        return df["close"] / df["close"].rolling(n).mean() - 1.0
    return f

def slope_tstat(n):
    def f(df):
        x = np.arange(n)
        xx = x - x.mean()
        def helper(y):
            if len(y) < n or np.isnan(y).any():
                return np.nan
            b = np.polyfit(x, y, 1)[0]
            resid = y - np.polyval(np.polyfit(x, y, 1), x)
            se = np.sqrt((resid ** 2).sum() / (n - 2)) / np.sqrt((xx ** 2).sum())
            return b / se if se > 0 else np.nan
        return df["close"].rolling(n).apply(helper, raw=True)
    return f

def vol_adj_rev(n):
    """Reversal scaled by local vol: -ret_n / rv(n)."""
    def f(df):
        r = df["close"].pct_change(n)
        vol = df["close"].pct_change().rolling(n).std() * np.sqrt(n)
        return -r / vol
    return f

def vol_z(n):
    """z-score of 1d return relative to rolling vol -> vol-scaled reversal."""
    def f(df):
        r = df["close"].pct_change()
        mu = r.rolling(n).mean()
        sd = r.rolling(n).std()
        return -(r - mu) / sd
    return f

def hl_pos(n):
    """Close position within trailing n-day high-low range (0..1)."""
    def f(df):
        hi = df["high"].rolling(n).max()
        lo = df["low"].rolling(n).min()
        return (df["close"] - lo) / (hi - lo)
    return f

def range_exp(n):
    """Today's range / trailing avg range -> range expansion signal."""
    def f(df):
        rng = df["high"] - df["low"]
        return rng / rng.rolling(n).mean()
    return f

def gap_rev(n):
    """N-day cumulative return skipping most recent day (trend component)."""
    def f(df):
        return np.log(df["close"] / df["close"].shift(n)) - np.log(df["close"] / df["close"].shift(1))
    return f

def rv_regime(short, long):
    """Vol regime: low short/long ratio = compressed vol (squeeze) -> sign-flipped."""
    return rv_ratio(short, long)

# ---------------------------------------------------------------- candidates
cands = {
    "rv5_over_rv20":      rv_ratio(5, 20),
    "rv10_over_rv60":     rv_ratio(10, 60),
    "rv5_over_rv60":      rv_ratio(5, 60),
    "parkinson_vol20":    parkinson(20),
    "semi_ratio20":       semi_ratio(20),
    "semi_ratio60":       semi_ratio(60),
    "skew20":             skew(20),
    "skew60":             skew(60),
    "downside_vol20":     downside_vol(20),
    "drawdown20":         drawdown(20),
    "drawdown60":         drawdown(60),
    "dd_speed20":         dd_speed(20),
    "ma_dist60":          ma_dist(60),
    "ma_dist120":         ma_dist(120),
    "slope_tstat20":      slope_tstat(20),
    "slope_tstat60":      slope_tstat(60),
    "vol_adj_rev5":       vol_adj_rev(5),
    "vol_z1_20":          vol_z(20),
    "hl_pos20":           hl_pos(20),
    "hl_pos60":           hl_pos(60),
    "range_exp20":        range_exp(20),
    "gap_rev20":          gap_rev(20),
    "gap_rev60":          gap_rev(60),
}

results = {}
for name, fn in cands.items():
    try:
        panel = factor_panel(closes, fn)
        cov = coverage(panel, closes)
        to = turnover(panel)
        ic1 = ic_analysis(panel, closes, fwd_days=1)
        ic5 = ic_analysis(panel, closes, fwd_days=5)
        dec = decay_analysis(panel, closes)
        results[name] = {"cov": cov, "turnover": to, "ic1": ic1, "ic5": ic5, "decay": dec}
        print(f"{name:16s} cov={cov:.3f} turn={to:.3f} | "
              f"IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit1={ic1['hit']:.3f} n={ic1['n_dates']} | "
              f"IC5={ic5['ic']:+.4f} | decay1d={dec[1]['ic']:+.4f} decay5d={dec[5]['ic']:+.4f} decay10d={dec[10]['ic']:+.4f}")
    except Exception as e:
        print(f"{name:16s} ERROR {e}")

print(f"\nelapsed {time.time()-T0:.1f}s")

# gate check
print("\n--- ADMISSION GATE (|IC|>=0.007, |ICIR|>=0.084 daily) ---")
for name, r in results.items():
    ic, icir = r["ic1"]["ic"], r["ic1"]["icir"]
    if abs(ic) >= 0.007 and abs(icir) >= 0.084 and r["cov"] >= 0.5:
        print(f"  PASS {name}: IC={ic:+.4f} ICIR={icir:+.3f} cov={r['cov']:.3f}")
