"""miner_3 batch 3: macro-structure / trend-quality / composite factors.

Explores signals orthogonal to plain momentum:
- rolling betas to macro (VIX, US10Y, DXY, SPX)
- monthly 12-1 momentum, trend R2, max drawdown, skew
- composite z-score momentum (noise reduction)
"""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
import numpy as np
from factor_harness import get_panels, evaluate, WATCH

closes, rets, ohlc, macro = get_panels()
mclose = macro  # macro panel already aligned to closes.index

factors = {}

# ---- rolling beta to macro changes (60d) ----
def roll_beta(y, x, n=60):
    """rolling beta of y on x using daily changes, aligned to closes.index."""
    dy = y.diff()
    dx = x.diff()
    cov = dy.rolling(n).cov(dx)
    var = dx.rolling(n).var()
    return cov / var

vix = mclose["VIX"]
us10y = mclose["US10Y"]
dxy = mclose["DXY"]
spx = closes["SPX"]

factors["vix_beta_60d"] = roll_beta(closes, vix)          # safe-haven: negative beta
factors["us10y_beta_60d"] = roll_beta(closes, us10y)      # rate sensitivity
factors["dxy_beta_60d"] = roll_beta(closes, dxy)          # USD sensitivity
factors["spx_beta_60d"] = roll_beta(closes, spx)          # global equity beta
factors["vix_beta_120d"] = roll_beta(closes, vix, 120)

# ---- monthly 12-1 momentum (skip recent 21d) ----
factors["mom_252_21"] = closes.pct_change(252) - closes.pct_change(21)
factors["mom_126_21"] = closes.pct_change(126) - closes.pct_change(21)

# ---- trend quality: R2 of linear fit over 60d ----
def trend_r2(px, n=60):
    x = np.arange(n)
    def r2(y):
        if len(y) < n or np.any(~np.isfinite(y)):
            return np.nan
        b = np.polyfit(x, y, 1)
        pred = np.polyval(b, x)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return px.rolling(n).apply(lambda y: r2(y.values), raw=False)
factors["trend_r2_60d"] = trend_r2(closes, 60)

# ---- max drawdown over 60d (negative = deeper drawdown) ----
def max_dd(px, n=60):
    def mdd(y):
        if len(y) < n or np.any(~np.isfinite(y)):
            return np.nan
        peak = np.maximum.accumulate(y)
        return float((y / peak - 1.0).min())
    return px.rolling(n).apply(lambda y: mdd(y.values), raw=False)
factors["maxdd_60d"] = max_dd(closes, 60)

# ---- skewness of 60d returns ----
factors["skew_60d"] = rets.rolling(60).skew()

# ---- vol ratio vol10/vol60 (vol trend) ----
rv10 = rets.rolling(10).std()
rv60 = rets.rolling(60).std()
factors["vol_ratio_10_60"] = rv10 / rv60 - 1.0

# ---- composite momentum z-scores ----
def zscore(px):
    return (px - px.mean(axis=1)) / px.std(axis=1)
mom20 = closes.pct_change(20)
mom60 = closes.pct_change(60)
mom120 = closes.pct_change(120)
mom180 = closes.pct_change(180)
z20, z60, z120, z180 = (zscore(m) for m in (mom20, mom60, mom120, mom180))
factors["mom_z_composite"] = (z20 + z60 + z120 + z180) / 4.0
factors["mom_z_60_180"] = (z60 + z120 + z180) / 3.0

print("=== BATCH 3 @ h=10 ===")
results = []
for name, f in factors.items():
    f = f.reindex(closes.index)
    results.append(evaluate(f, rets, h=10, name=name, verbose=True))

print("\n=== BATCH 3 @ h=5 ===")
for name, f in factors.items():
    f = f.reindex(closes.index)
    evaluate(f, rets, h=5, name=name, verbose=True)

print("\n=== PASS GATE (|IC|>=0.007 & |ICIR|>=0.084 @ h=10) ===")
for r in results:
    if abs(r["mean_ic"]) >= 0.007 and abs(r["icir"]) >= 0.084:
        print(f"PASS {r['name']}: IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.3f} t={r['tstat']:+.1f}")
