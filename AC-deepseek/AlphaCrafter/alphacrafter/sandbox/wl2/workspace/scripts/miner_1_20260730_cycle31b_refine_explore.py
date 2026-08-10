"""miner_1 cycle 31b: refine macro-conditional family + skew/vol variants.

From cycle31: eurusd collinear w/ dxy factor; usdjpy/btc/us10y close but weak ICIR;
skew20 near gate. Explore:
  A) skew_60 (longer-window skew), inv_vol20 (calm proxy check vs calmness_20)
  B) cn10y_beta_cond, wti_beta_cond, xau_beta_cond (China rates / energy / gold drivers)
  C) vix_timing_beta_60 (SPX beta x -20d VIX change: defensive rotation on vol spikes)
  D) usdjpy_beta_cond_120x60 (longer beta/mom windows)
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, VISIBLE_THROUGH, load_asset, load_panel,
                         macro_series, per_asset, forward_returns, compute_ic,
                         validate_factor)

frames = {a: load_asset(a) for a in TRADABLES}
close_panel = load_panel()

def beta_cond(asset_close, driver_close, w=60, m=20, minp_frac=0.5):
    dcs = driver_close.reindex(asset_close.index).ffill()
    ar = asset_close.pct_change()
    dr = dcs.pct_change()
    df = pd.concat([ar.rename("a"), dr.rename("d")], axis=1).dropna()
    minp = max(int(w * minp_frac), 15)
    cov = df["a"].rolling(w, min_periods=minp).cov(df["d"])
    var = df["d"].rolling(w, min_periods=minp).var()
    beta = cov / var
    mom = dcs / dcs.shift(m) - 1.0
    return beta * mom.reindex(beta.index)

def per_asset_beta_cond(close_panel, driver, **kw):
    return per_asset(close_panel, beta_cond, driver, **kw)

# ---- library signals (recomputed) ----
def mom20_volproxy60(s):
    mom = s.shift(5) / s.shift(25) - 1.0
    proxy = s.shift(5) / s.shift(65) - 1.0
    return mom / (1.0 + proxy.abs())

def calmness_20(s):
    r = s.pct_change()
    v = r.rolling(20).std()
    return r.abs().rolling(20).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan,
        raw=True)

open_panel = pd.concat(
    [pd.Series(frames[a]["open"].astype(float).values,
               index=pd.to_datetime(frames[a]["date"]), name=a) for a in TRADABLES],
    axis=1).sort_index()

lib = {}
lib["mom20_volproxy60"] = per_asset(close_panel, mom20_volproxy60)
lib["dxy_beta_cond_60x20"] = per_asset_beta_cond(close_panel, macro_series("DXY"))
lib["calmness_20"] = per_asset(close_panel, calmness_20)
lib["intraday_drift_20"] = per_asset(
    close_panel, lambda s: (s / open_panel[s.name] - 1.0).rolling(20, min_periods=10).mean())

# ---- candidates ----
cands = {}
cands["skew_60"] = per_asset(close_panel, lambda s: s.pct_change().rolling(60, min_periods=30).skew())
cands["inv_vol20"] = per_asset(close_panel, lambda s: 1.0 / s.pct_change().rolling(20, min_periods=10).std())
cands["cn10y_beta_cond_60x20"] = per_asset_beta_cond(close_panel, close_panel["CN10Y"])
cands["wti_beta_cond_60x20"] = per_asset_beta_cond(close_panel, close_panel["WTI"])
cands["xau_beta_cond_60x20"] = per_asset_beta_cond(close_panel, close_panel["XAU"])
cands["usdjpy_beta_cond_120x60"] = per_asset_beta_cond(close_panel, macro_series("USDJPY"), w=120, m=60)

def vix_timing_beta(asset_close, spx_close, vix_close, w=60):
    sp = spx_close.reindex(asset_close.index).ffill()
    vx = vix_close.reindex(asset_close.index).ffill()
    ar = asset_close.pct_change()
    sr = sp.pct_change()
    df = pd.concat([ar.rename("a"), sr.rename("s")], axis=1).dropna()
    minp = max(int(w * 0.5), 15)
    beta = df["a"].rolling(w, min_periods=minp).cov(df["s"]) / df["s"].rolling(w, min_periods=minp).var()
    dV = vx / vx.shift(20) - 1.0
    return beta * (-dV.reindex(beta.index))

cands["vix_timing_beta_60"] = per_asset(
    close_panel, lambda s: vix_timing_beta(s, close_panel["SPX"], macro_series("VIX"), 60))

# ---- validate ----
fwd_cache = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd_cache[str(h)] = forward_returns(close_panel, h)

print("=" * 112)
print(f"CYCLE 31b: macro-conditional refine + skew/vol | visible through {VISIBLE_THROUGH}")
print("=" * 112)
results = {}
for name, sig in cands.items():
    m = validate_factor(sig, close_panel, library=lib, fwd_cache=fwd_cache)
    ic, icir = abs(m["ic"]), abs(m["icir"])
    passed = (ic >= 0.007) and (icir >= 0.084) and (m.get("max_abs_library_correlation", 1.0) < 0.5)
    results[name] = {"metrics": m, "pass": passed}
    print(f"[{name}] IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']} "
          f"turn={m['turnover_10_rank']} maxlibcorr={m.get('max_abs_library_correlation')} "
          f"=> {'PASS' if passed else 'fail'}")
    print(f"    libcorr={m.get('library_pairwise_corr')}")

print("\n=== REGIME BREAKDOWN (10d IC by sub-period) ===")
for name, sig in cands.items():
    ic_ser = compute_ic(sig, fwd_cache["10"]).dropna()
    parts = []
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 30:
            sd = sub.std()
            parts.append(f"{r0[:4]}-{r1[:4]}: ic={sub.mean():+.4f} icir={sub.mean()/sd if sd>0 else 0:+.3f} n={len(sub)}")
    print(f"  {name:26s} | " + " | ".join(parts))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/_miner1_cycle31b_explore_results.json", "w"), indent=1, default=float)
print("\nDONE cycle31b exploration")
