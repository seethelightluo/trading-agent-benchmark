"""miner_3 batch 4: fresh factor ideas - liquidity, autocorrelation, conditional
macro, trend-quality, downside beta, relative strength. All on 15-asset universe.

Checks the admission gate |IC|>=0.007 & |ICIR|>=0.084 at h=10 (and h=5/20 decay view).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_harness import get_panels, evaluate, WATCH

closes, rets, ohlc, macro = get_panels()
vix = macro["VIX"]
dxy = macro["DXY"]
us10y = closes["US10Y"]
spx = closes["SPX"]

# ---- volume panel (may be missing for some assets) ----
vol_panel = pd.concat({a: ohlc[a]["volume"] for a in WATCH}, axis=1).reindex(closes.index)

factors = {}
rv20 = rets.rolling(20).std()
rv60 = rets.rolling(60).std()

# 1. Kaufman efficiency ratio 60d (trend quality, low turnover)
def eff_ratio(px, n=60):
    move = (px - px.shift(n)).abs()
    path = px.diff().abs().rolling(n).sum()
    return (move / path).replace([np.inf, -np.inf], np.nan)
factors["eff_ratio_60d"] = eff_ratio(closes, 60)

# 2. Amihud illiquidity 20d (|ret|/volume), negated -> liquidity factor
def amihud(px, vol, n=20):
    illiq = (px.pct_change().abs() / vol.replace(0, np.nan)).rolling(n).mean()
    return -illiq  # high = liquid
factors["amihud_liquidity_20d"] = amihud(closes, vol_panel, 20)

# 3. Volume trend 20/60 (volume expansion)
vol20 = vol_panel.rolling(20).mean()
vol60 = vol_panel.rolling(60).mean()
factors["vol_trend_20_60"] = (vol20 / vol60 - 1.0).replace([np.inf, -np.inf], np.nan)

# 4. Return autocorrelation 20d (sign persistence, negated: mean reversion)
def autocorr(px, n=20):
    r = px.pct_change()
    out = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for a in px.columns:
        ra = r[a]
        out[a] = ra.rolling(n).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) >= 3 else np.nan, raw=True)
    return out
factors["ret_autocorr_20d"] = autocorr(closes, 20)

# 5. Vol-adjusted momentum: mom60 / rv20 (Sharpe-like trend)
factors["ts_mom_60_20"] = closes.pct_change(60) / rv20

# 6. Cross-sectional relative momentum 30d (asset vs cross-section mean)
cs_mean_30 = closes.pct_change(30).mean(axis=1)
factors["rel_mom_30d"] = closes.pct_change(30).sub(cs_mean_30, axis=0)

# 7. Downside beta to SPX (60d, only days SPX<0)
def downside_beta(panel, mkt, n=60):
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    rm = mkt.pct_change()
    for a in panel.columns:
        ra = panel[a].pct_change()
        cov = ra.where(rm < 0).rolling(n).cov(rm.where(rm < 0))
        var = rm.where(rm < 0).rolling(n).var()
        out[a] = cov / var
    return out
factors["downside_beta_60d"] = downside_beta(closes, spx, 60)

# 8. Conditional DXY factor: dxy beta * 20d DXY move
def roll_beta(panel, x, n=60):
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    dx = x.diff()
    for a in panel.columns:
        y = panel[a].diff()
        out[a] = y.rolling(n).cov(dx) / dx.rolling(n).var()
    return out
dxy_beta = roll_beta(closes, dxy, 60)
factors["dxy_cond_60x20"] = dxy_beta.mul((dxy / dxy.shift(20) - 1.0).reindex(closes.index), axis=0)

# 9. Conditional US10Y factor: us10y beta * 20d yield move
u10_beta = roll_beta(closes, us10y, 60)
factors["us10y_cond_60x20"] = u10_beta.mul((us10y / us10y.shift(20) - 1.0).reindex(closes.index), axis=0)

# 10. Trend t-stat (60d OLS slope / se) - trend strength with noise control
def trend_tstat(px, n=60):
    x = np.arange(n)
    out = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for a in px.columns:
        def tstat(y):
            if len(y) < n or np.any(~np.isfinite(y)):
                return np.nan
            b = np.polyfit(x, y, 1)
            resid = y - np.polyval(b, x)
            se = np.sqrt(np.sum(resid ** 2) / (n - 2)) / np.sqrt(np.sum((x - x.mean()) ** 2))
            return b[0] / se if se > 0 else np.nan
        out[a] = px[a].rolling(n).apply(lambda y: tstat(y.values), raw=False)
    return out
factors["trend_tstat_60d"] = trend_tstat(closes, 60)

# 11. Distance from 52-week high (240d)
factors["dist_240d_high"] = closes / closes.rolling(240).max() - 1.0

# 12. Range ratio 20d: mean(high-low)/close (vol proxy, additive info vs rv)
def range_ratio(px, o, n=20):
    rr = pd.concat({a: (o[a]["high"] - o[a]["low"]) / px[a] for a in WATCH}, axis=1).reindex(px.index)
    return rr.rolling(n).mean()
factors["range_ratio_20d"] = range_ratio(closes, ohlc, 20)

print("=== BATCH 4 @ h=10 ===")
results = []
for name, f in factors.items():
    f = f.reindex(closes.index)
    results.append(evaluate(f, rets, h=10, name=name, verbose=True))

print("\n=== BATCH 4 @ h=5 ===")
for name, f in factors.items():
    f = f.reindex(closes.index)
    evaluate(f, rets, h=5, name=name, verbose=True)

print("\n=== BATCH 4 @ h=20 ===")
for name, f in factors.items():
    f = f.reindex(closes.index)
    evaluate(f, rets, h=20, name=name, verbose=True)

print("\n=== PASS GATE (|IC|>=0.007 & |ICIR|>=0.084 @ h=10) ===")
for r in results:
    if abs(r["mean_ic"]) >= 0.007 and abs(r["icir"]) >= 0.084:
        print(f"PASS {r['name']}: IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.3f} t={r['tstat']:+.1f} hit={r['hit']:.2f}")
