"""Cycle 30 Family A: trend-quality / path-consistency factors.

Active library (2): mom20_volproxy60 (raw momentum magnitude, vol-damped),
dxy_beta_cond_60x20 (macro beta x DXY momentum).
Goal: find candidates from trend QUALITY (not magnitude): R2 of price path,
autocorrelation-conditioned momentum, drawup/drawdown asymmetry, Sharpe-style
mom/vol, calmness. All at 10d admission horizon, IC>=0.007, |ICIR|>=0.084,
|rho|<0.5 vs both active factors.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, macro_series, per_asset,
                         forward_returns, validate_factor)

panel = load_panel()
close = panel

# ---- Rebuild 2 ACTIVE library signals exactly ----
mom60_proxy = per_asset(close, lambda s: s.shift(5) / s.shift(65) - 1.0)
damp = 1.0 / (1.0 + mom60_proxy.abs())
mom20_raw = per_asset(close, lambda s: s.shift(5) / s.shift(25) - 1.0)
sig_mom = mom20_raw * damp

dxy = macro_series("DXY")
dxy_ret = dxy.pct_change()
dxy_20 = dxy / dxy.shift(20) - 1.0
beta_parts = {}
for a in close.columns:
    s = close[a].dropna()
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), dxy_ret.reindex(ar.index).rename("d")], axis=1).dropna()
    b = df["a"].rolling(60).cov(df["d"]) / df["d"].rolling(60).var()
    beta_parts[a] = b.reindex(panel.index)
beta_panel = pd.DataFrame(beta_parts, index=panel.index)
sig_dxy = beta_panel.mul(dxy_20.reindex(beta_panel.index), axis=0)

library = {"mom20_volproxy60": sig_mom, "dxy_beta_cond_60x20": sig_dxy}

# ---- Candidate builders (per-asset own calendar) ----
def r2_trend(s, w=60):
    """R^2 of log-price vs time over trailing w days (trend consistency)."""
    def _r2(x):
        if len(x) < 30 or np.std(x) == 0:
            return np.nan
        t = np.arange(len(x))
        return float(np.corrcoef(x, t)[0, 1] ** 2)
    return np.log(s).rolling(w).apply(_r2, raw=True)

def mom_autocorr_cond(s, w=20, skip=5):
    """20d momentum (skip5) x (1 + autocorr of daily returns over 20d)."""
    mom = s.shift(skip) / s.shift(skip + w) - 1.0
    r = s.pct_change()
    ac = r.rolling(w).apply(lambda x: pd.Series(x).autocorr(1) if len(x) >= 10 else np.nan, raw=False)
    return mom * (1.0 + ac)

def drawup_drawdown_ratio(s, w=60):
    """Ratio of max drawup magnitude to max drawdown magnitude over w days."""
    def _ratio(x):
        if len(x) < 30:
            return np.nan
        c = np.asarray(x, dtype=float)
        c = c / c[0]
        dd = float(np.min(c / np.maximum.accumulate(c) - 1.0))   # negative
        du = float(np.max(c / np.minimum.accumulate(c) - 1.0))   # positive
        if abs(dd) < 1e-12:
            return np.nan
        return du / abs(dd)
    return s.rolling(w).apply(_ratio, raw=True)

def sharpe_mom_20_skip5(s):
    """20d momentum (skip5) divided by 20d daily-return vol (Sharpe-style, no abs-damp)."""
    mom = s.shift(5) / s.shift(25) - 1.0
    vol = s.pct_change().rolling(20).std()
    return mom / vol

def calmness_20(s):
    """Fraction of last 20d with |daily ret| < 0.5 * 20d std (quiet persistence)."""
    r = s.pct_change()
    v = r.rolling(20).std()
    calm = r.abs().rolling(20).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan, raw=True)
    return calm

def r2_trend_sign(s, w=60):
    """R^2 of log-price vs time x sign of slope (trend quality with direction)."""
    def _r2s(x):
        if len(x) < 30:
            return np.nan
        t = np.arange(len(x))
        c = np.corrcoef(x, t)
        return float(c[0, 1] ** 2 * np.sign(c[0, 1]))
    return np.log(s).rolling(w).apply(_r2s, raw=True)

# ---- Build candidates ----
cands = {}
cands["r2_trend_60"] = per_asset(close, r2_trend, 60)
cands["r2_trend_60_signed"] = per_asset(close, r2_trend_sign, 60)
cands["mom_autocorr_cond_20"] = per_asset(close, mom_autocorr_cond, 20, 5)
cands["drawup_drawdown_ratio_60"] = per_asset(close, drawup_drawdown_ratio, 60)
cands["sharpe_mom_20_skip5"] = per_asset(close, sharpe_mom_20_skip5)
cands["calmness_20"] = per_asset(close, calmness_20)

# ---- Validate ----
fwd_cache = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd_cache[str(h)] = forward_returns(panel, h)

print("=" * 110)
print("CYCLE 30 FAMILY A: trend-quality / path consistency  |  visible through 2026-07-29")
print("universe:", len(close.columns), "assets | panel dates:", len(panel))
print("=" * 110)
for name, sig in cands.items():
    m = validate_factor(sig, panel, library=library, fwd_cache=fwd_cache)
    ic, icir = abs(m["ic"]), abs(m["icir"])
    passed = (ic >= 0.007) and (icir >= 0.084)
    print(f"[{name}] IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} "
          f"n={m['n_ic_dates']} cov_asset={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10_rank']} maxlibcorr={m['max_abs_library_correlation']} "
          f"decay={m['decay_ic_by_horizon']} => {'PASS' if passed else 'fail'}")
    print(f"    libcorr={m.get('library_pairwise_corr')}")
    print()
