"""miner_1 research cycle 2026-07-16: screen decorrelated risk/vol/trend factor families (fast mode, v2).

Library is dominated by 1-3d reversal variants + one 10d-skip5 momentum. Focus here is
families built on volatility structure, skew/risk asymmetry, drawdown and trend that are
likely decorrelated from the reversal cluster.

Fast mode: only IC1 + IC5 computed, no full decay for all candidates.
Admission gates (daily): |IC| >= 0.0070, |ICIR| >= 0.0840.
Vectorized slope t-stat (no rolling.apply).
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
print(f"loaded {len(closes)} symbols; data through {CUT.date()}", flush=True)
fwd1 = build_returns(closes, 1)
fwd5 = build_returns(closes, 5)

# ---------------------------------------------------------------- factor fns
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
        return hl.rolling(n).std() * np.sqrt(252)
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

def drawdown(n):
    def f(df):
        return df["close"] / df["close"].rolling(n).max() - 1.0
    return f

def dd_speed(n):
    def f(df):
        dd = df["close"] / df["close"].rolling(n).max() - 1.0
        return dd.diff()
    return f

def ma_dist(n):
    def f(df):
        return df["close"] / df["close"].rolling(n).mean() - 1.0
    return f

def slope_tstat_vec(n):
    """Vectorized OLS t-stat of close on time over rolling window n.
    Correct window-start handling: slope = Sxy/Sxx with x local to the window.
    Uses rolling sums of t, t^2, y, t*y (t = global integer time)."""
    def f(df):
        y = df["close"].values.astype(float)
        m = np.isnan(y)
        y = np.where(m, np.nan, y)
        t = np.arange(len(y), dtype=float)
        t2 = t * t
        sy  = pd.Series(y).rolling(n).sum().values
        st  = pd.Series(t).rolling(n).sum().values
        st2 = pd.Series(t2).rolling(n).sum().values
        sty = pd.Series(t * y).rolling(n).sum().values
        n_valid = pd.Series(~m).rolling(n).sum().values
        with np.errstate(all="ignore"):
            # centered sums for window [s, s+n-1]
            tbar = st / n
            ybar = sy / n
            sxx = st2 - st * st / n          # sum (t - tbar)^2
            sxy = sty - st * sy / n          # sum (t - tbar)(y - ybar)
            b = sxy / sxx
            syy = pd.Series(y * y).rolling(n).sum().values - n * ybar * ybar
            sse = syy - b * sxy
            mse = sse / (n - 2)
            se_b = np.sqrt(mse / sxx)
            tstat = np.where(se_b > 0, b / se_b, np.nan)
            tstat = np.where(n_valid < n, np.nan, tstat)
            tstat = np.where(sxx <= 0, np.nan, tstat)
        return pd.Series(tstat, index=df.index)
    return f

def vol_adj_rev(n):
    def f(df):
        r = df["close"].pct_change(n)
        vol = df["close"].pct_change().rolling(n).std() * np.sqrt(n)
        return -r / vol
    return f

def vol_z(n):
    def f(df):
        r = df["close"].pct_change()
        mu = r.rolling(n).mean()
        sd = r.rolling(n).std()
        return -(r - mu) / sd
    return f

def hl_pos(n):
    def f(df):
        hi = df["high"].rolling(n).max()
        lo = df["low"].rolling(n).min()
        return (df["close"] - lo) / (hi - lo)
    return f

def range_exp(n):
    def f(df):
        rng = df["high"] - df["low"]
        return rng / rng.rolling(n).mean()
    return f

def gap_rev(n):
    def f(df):
        return np.log(df["close"] / df["close"].shift(n)) - np.log(df["close"] / df["close"].shift(1))
    return f

# ---------------------------------------------------------------- candidates (reduced, focused)
cands = {
    "rv5_over_rv20":      rv_ratio(5, 20),
    "rv10_over_rv60":     rv_ratio(10, 60),
    "parkinson_vol20":    parkinson(20),
    "semi_ratio20":       semi_ratio(20),
    "semi_ratio60":       semi_ratio(60),
    "skew20":             skew(20),
    "skew60":             skew(60),
    "drawdown20":         drawdown(20),
    "drawdown60":         drawdown(60),
    "dd_speed20":         dd_speed(20),
    "ma_dist60":          ma_dist(60),
    "ma_dist120":         ma_dist(120),
    "slope_tstat20":      slope_tstat_vec(20),
    "slope_tstat60":      slope_tstat_vec(60),
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
    t1 = time.time()
    try:
        panel = factor_panel(closes, fn)
        cov = coverage(panel, closes)
        to = turnover(panel)
        ic1 = ic_analysis(panel, closes, fwd_days=1)
        ic5 = ic_analysis(panel, closes, fwd_days=5)
        results[name] = {"cov": cov, "turnover": to, "ic1": ic1, "ic5": ic5}
        print(f"{name:16s} cov={cov:.3f} turn={to:.3f} | "
              f"IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit1={ic1['hit']:.3f} n={ic1['n_dates']} | "
              f"IC5={ic5['ic']:+.4f} ({time.time()-t1:.1f}s)", flush=True)
    except Exception as e:
        print(f"{name:16s} ERROR {e}", flush=True)

print(f"\nelapsed {time.time()-T0:.1f}s", flush=True)

# gate check
print("\n--- ADMISSION GATE (|IC|>=0.007, |ICIR|>=0.084 daily) ---")
passers = {}
for name, r in results.items():
    ic, icir = r["ic1"]["ic"], r["ic1"]["icir"]
    if abs(ic) >= 0.007 and abs(icir) >= 0.084 and r["cov"] >= 0.5:
        passers[name] = r
        print(f"  PASS {name}: IC={ic:+.4f} ICIR={icir:+.3f} cov={r['cov']:.3f} turn={r['turnover']:.3f}")

if passers:
    import pickle
    with open("scripts/_miner1_passers_riskvol.pkl", "wb") as fh:
        pickle.dump({k: {kk: results[k][kk] for kk in ["cov","turnover","ic1","ic5"]}
                     for k in passers}, fh)
    print("saved scripts/_miner1_passers_riskvol.pkl:", list(passers.keys()))
