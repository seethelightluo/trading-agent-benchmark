"""miner_3 2026-07-30 -- Screen round 2: fresh orthogonal factor families.

Library currently holds price momentum (mom_10d_skip5) and regime-beta
factors (vix_beta_cond_60x20, yield_beta_cond_60x20). This screen targets
families orthogonal to those: intraday/overnight decomposition, volume flow,
vol term structure, downside-vol asymmetry, range-ratio, distributional
(skew/kurt), and cross-asset beta to BTC / XAU.

SCREEN ONLY (no persistence). The best non-degenerate candidate that clears
IC/ICIR gates and the library-correlation gate will get a deep-validation
script afterwards.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, "scripts")
from miner_3_20260730_common import (
    load_data, factor_ic_table, coverage_stats, rank_turnover,
)

data = load_data(days=3200)
closes = {a: d["close"].astype(float) for a, d in data.items()}
opens = {a: d["open"].astype(float) for a, d in data.items()}
highs = {a: d["high"].astype(float) for a, d in data.items()}
lows = {a: d["low"].astype(float) for a, d in data.items()}
vols = {a: d["volume"].astype(float).replace(0, np.nan) for a, d in data.items()}
print(f"[screen2] assets={len(data)} range={min(c.index.min() for c in closes.values()).date()}..{max(c.index.max() for c in closes.values()).date()}")

# ---- candidate factor builders (per-asset series) ----
def overnight_ret_20(c, o, **kw):
    return (o / c.shift(1) - 1.0).rolling(20).mean()

def intraday_ret_20(c, o, **kw):
    return (c / o - 1.0).rolling(20).mean()

def vol_flow_20(c, o, h, l, v, **kw):
    d = c.diff()
    return (v * np.sign(d)).rolling(20).sum() / v.rolling(20).sum()

def vol_ratio_5x60(c, o, h, l, v, **kw):
    return v.rolling(5).mean() / v.rolling(60).mean() - 1.0

def vol_term_10x60(c, **kw):
    r = c.pct_change()
    return r.rolling(10).std() / r.rolling(60).std() - 1.0

def downside_vol_ratio_60(c, **kw):
    r = c.pct_change()
    dneg = r.clip(upper=0.0)
    return dneg.rolling(60).std() / r.rolling(60).std()

def range_vol_ratio_20(c, o, h, l, **kw):
    rng = ((h - l) / c).rolling(20).mean()
    rv = c.pct_change().rolling(20).std()
    return rng / rv.replace(0, np.nan)

def skew_20(c, **kw):
    return c.pct_change().rolling(20).skew()

def kurt_60(c, **kw):
    return c.pct_change().rolling(60).kurt()

def obv_slope_20(c, o, h, l, v, **kw):
    obv = (np.sign(c.diff()) * v).fillna(0.0).cumsum()
    # linear slope of OBV over 20d, normalized by rolling mean abs volume
    x = np.arange(20, dtype=float)
    s = obv.rolling(20).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
    return s / v.rolling(20).mean()

def high_low_pos_20(c, o, h, l, **kw):
    rng = (h - l).replace(0, np.nan)
    return ((c - l) / rng).rolling(20).mean()

def beta_to(factor_close):
    def _f(c, **kw):
        r = c.pct_change()
        fr = factor_close.pct_change()
        beta = r.rolling(60).cov(fr) / fr.rolling(60).var()
        move = factor_close / factor_close.shift(20) - 1.0
        return -beta * move
    return _f

btc_close = closes["BTC"]
xau_close = closes["XAU"]
btc_beta_cond = beta_to(btc_close)
xau_beta_cond = beta_to(xau_close)

CANDIDATES = [
    ("overnight_ret_20", "mean overnight gap (open vs prev close), 20d", overnight_ret_20),
    ("intraday_ret_20", "mean intraday return (close vs open), 20d", intraday_ret_20),
    ("vol_flow_20", "volume-weighted return direction (OBV-style flow), 20d", vol_flow_20),
    ("vol_ratio_5x60", "volume surge: 5d/60d mean volume - 1", vol_ratio_5x60),
    ("vol_term_10x60", "vol term structure: rv10/rv60 - 1", vol_term_10x60),
    ("downside_vol_ratio_60", "downside semi-deviation / total vol, 60d", downside_vol_ratio_60),
    ("range_vol_ratio_20", "mean daily range / realized vol, 20d", range_vol_ratio_20),
    ("skew_20", "rolling skewness of 20d returns", skew_20),
    ("kurt_60", "rolling excess kurtosis of 60d returns", kurt_60),
    ("obv_slope_20", "20d linear slope of on-balance volume (normalized)", obv_slope_20),
    ("high_low_pos_20", "close location in daily range, 20d mean", high_low_pos_20),
    ("btc_beta_cond_60x20", "-beta(asset,BTC,60d) * BTC 20d move", btc_beta_cond),
    ("xau_beta_cond_60x20", "-beta(asset,XAU,60d) * XAU 20d move", xau_beta_cond),
]

# replicate library factors incl. yield_beta for correlation reference
def lib_panels():
    lib = {}
    for a, c in closes.items():
        lib.setdefault("mom_10d_skip5", {})[a] = c.shift(5) / c.shift(15) - 1.0
        lib.setdefault("mom_120d_skip5", {})[a] = c.shift(5) / c.shift(125) - 1.0
        lib.setdefault("vol_of_vol20x60", {})[a] = c.pct_change().rolling(20).std().rolling(60).std()
    # vix beta
    try:
        from alphacrafter.sim.utils import get_stock_daily_data
        vdf = get_stock_daily_data(symbol="VIX", days=3200)
        vix = vdf.set_index(pd.to_datetime(vdf["date"])).sort_index()["close"].astype(float)
        for a, c in closes.items():
            r = c.pct_change()
            beta = r.rolling(60).cov(vix.pct_change()) / vix.pct_change().rolling(60).var()
            move = vix / vix.shift(20) - 1.0
            lib.setdefault("vix_beta_cond_60x20", {})[a] = -beta * move
    except Exception as e:
        print("vix load fail", e)
    # yield beta (US10Y)
    us10 = closes["US10Y"]
    for a, c in closes.items():
        r = c.pct_change()
        yr = us10.pct_change()
        beta = r.rolling(60).cov(yr) / yr.rolling(60).var()
        move = us10 / us10.shift(20) - 1.0
        lib.setdefault("yield_beta_cond_60x20", {})[a] = -beta * move
    return lib

LIB = lib_panels()

def lib_corr(factor):
    fdf = pd.DataFrame(factor).stack()
    fdf = fdf[fdf.notna()]
    out = {}
    for fid, lf in LIB.items():
        ldf = pd.DataFrame(lf).stack()
        both = fdf.index.intersection(ldf.index)
        if len(both) < 100:
            out[fid] = float("nan")
            continue
        rho, _ = pearsonr(fdf.loc[both].values, ldf.loc[both].values)
        out[fid] = rho
    vals = [abs(v) for v in out.values() if np.isfinite(v)]
    return (max(vals) if vals else float("nan")), out

results = {}
for fid, desc, fn in CANDIDATES:
    try:
        f = {a: fn(c, o=opens[a], h=highs[a], l=lows[a], v=vols[a]) for a, c in closes.items()}
    except Exception as e:
        print(f"[screen2] {fid}: BUILD ERROR {e}")
        continue
    f = {a: s.replace([np.inf, -np.inf], np.nan) for a, s in f.items()}
    tbl = factor_ic_table(f, data, horizons=(1, 3, 5, 10, 20), min_assets=8, primary_h=10)
    prim = tbl[10]
    if prim is None:
        print(f"[screen2] {fid:24s} DEGENERATE (no valid IC dates)")
        results[fid] = None
        continue
    cov = coverage_stats(f, data)
    to = rank_turnover(f)
    maxrho, rho_map = lib_corr(f)
    gate_ic = abs(prim["ic"]) >= 0.0070
    gate_icir = abs(prim["icir"]) >= 0.0840
    gate_rho = maxrho < 0.5 if np.isfinite(maxrho) else False
    flag = "PASS" if (gate_ic and gate_icir and gate_rho) else "fail"
    print(f"[screen2] {fid:24s} ic10={prim['ic']:+.4f} icir10={prim['icir']:+.4f} "
          f"hit={prim['ic_hit']:.3f} n={prim['n_dates']:4d} cov={cov['coverage_asset_days']:.3f} "
          f"turn={to:5.2f} maxrho={maxrho:.3f} decay10/1={tbl[10]['ic']:.4f}/{tbl[1]['ic'] if tbl[1] else float('nan'):.4f} -> {flag}")
    results[fid] = dict(
        ic=prim["ic"], icir=prim["icir"], hit=prim["ic_hit"], n=prim["n_dates"],
        cov=cov["coverage_asset_days"], turn=to, maxrho=(round(maxrho, 4) if np.isfinite(maxrho) else None),
        rho_map={k: round(v, 3) for k, v in rho_map.items()},
        decay={h: (round(v["ic"], 4) if v else None) for h, v in tbl.items()},
    )

print("\n=== SUMMARY (primary horizon 10d, gate IC>=0.007 ICIR>=0.084 rho<0.5) ===")
for fid, r in results.items():
    if r is None:
        print(f"  {fid:24s} DEGENERATE")
    else:
        print(f"  {fid:24s} ic={r['ic']:+.4f} icir={r['icir']:+.4f} n={r['n']} maxrho={r['maxrho']}")
