"""miner_1 cycle 31: macro-conditional beta x momentum + path-structure candidates.

Goal: find factors orthogonal to active library
  (mom20_volproxy60, dxy_beta_cond_60x20, calmness_20, intraday_drift_20).

Families:
  A) macro-conditional: rolling 60d beta of asset ret on macro-driver ret
     x 20d driver momentum  (DXY sibling already in library; test USDJPY, EURUSD,
     USDCNY, US10Y, BTC as drivers).
  B) path structure: overnight gap persistence, signed return skew, ATR-range norm.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, MACRO, VISIBLE_THROUGH, load_asset, load_panel,
                         macro_series, per_asset, forward_returns, compute_ic,
                         validate_factor, report)

# ------------------------------------------------------------------------------
# OHLC panels (open, high, low, close) on union index
# ------------------------------------------------------------------------------
frames = {a: load_asset(a) for a in TRADABLES}
close_panel = load_panel()
open_panel = pd.concat(
    [pd.Series(frames[a]["open"].astype(float).values,
               index=pd.to_datetime(frames[a]["date"]), name=a) for a in TRADABLES],
    axis=1).sort_index()
high_panel = pd.concat(
    [pd.Series(frames[a]["high"].astype(float).values,
               index=pd.to_datetime(frames[a]["date"]), name=a) for a in TRADABLES],
    axis=1).sort_index()
low_panel = pd.concat(
    [pd.Series(frames[a]["low"].astype(float).values,
               index=pd.to_datetime(frames[a]["date"]), name=a) for a in TRADABLES],
    axis=1).sort_index()

print("panel dates:", len(close_panel), "| assets:", close_panel.shape[1])

# ------------------------------------------------------------------------------
# helper: rolling beta of asset series on a driver series x driver momentum
# ------------------------------------------------------------------------------
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
    return (beta * mom.reindex(beta.index))

def per_asset_beta_cond(close_panel, driver):
    return per_asset(close_panel, beta_cond, driver)

# ------------------------------------------------------------------------------
# library signals (recomputed, aligned)
# ------------------------------------------------------------------------------
def mom20_volproxy60(s):
    mom = s.shift(5) / s.shift(25) - 1.0
    proxy = s.shift(5) / s.shift(65) - 1.0
    return mom / (1.0 + proxy.abs())

def calmness_20(s):
    r = s.pct_change()
    v = r.rolling(20).std()
    out = r.abs().rolling(20).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan,
        raw=True)
    return out

def intraday_drift_20(close_s, open_s):
    return (close_s / open_s - 1.0).rolling(20, min_periods=10).mean()

lib = {}
lib["mom20_volproxy60"] = per_asset(close_panel, mom20_volproxy60)
dxy = macro_series("DXY")
lib["dxy_beta_cond_60x20"] = per_asset_beta_cond(close_panel, dxy)
lib["calmness_20"] = per_asset(close_panel, calmness_20)
lib["intraday_drift_20"] = per_asset(
    close_panel, lambda s: intraday_drift_20(s, open_panel[s.name]))
print("library factors:", list(lib.keys()))

# ------------------------------------------------------------------------------
# candidates
# ------------------------------------------------------------------------------
cands = {}
# Family A: macro-conditional
for drv, name in [("USDJPY", "usdjpy_beta_cond_60x20"),
                  ("EURUSD", "eurusd_beta_cond_60x20"),
                  ("USDCNY", "usdcny_beta_cond_60x20")]:
    cands[name] = per_asset_beta_cond(close_panel, macro_series(drv))

# US10Y as in-universe rate driver
cands["us10y_beta_cond_60x20"] = per_asset_beta_cond(close_panel, close_panel["US10Y"])
# BTC as in-universe risk-appetite driver
cands["btc_beta_cond_60x20"] = per_asset_beta_cond(close_panel, close_panel["BTC"])

# Family B: path structure
def overnight_gap(close_s, open_s):
    op = open_s.reindex(close_s.index).ffill()
    return (op / close_s.shift(1) - 1.0).rolling(20, min_periods=10).mean()

cands["overnight_gap_20"] = per_asset(
    close_panel, lambda s: overnight_gap(s, open_panel[s.name]))
cands["skew20"] = per_asset(
    close_panel, lambda s: s.pct_change().rolling(20, min_periods=10).skew())
def atr_range_norm(close_s, high_s, low_s):
    hp = high_s.reindex(close_s.index).ffill()
    lp = low_s.reindex(close_s.index).ffill()
    return ((hp - lp) / close_s).rolling(20, min_periods=10).mean()

cands["atr_range_norm_20"] = per_asset(
    close_panel, lambda s: atr_range_norm(s, high_panel[s.name], low_panel[s.name]))

# ------------------------------------------------------------------------------
# validate
# ------------------------------------------------------------------------------
fwd_cache = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd_cache[str(h)] = forward_returns(close_panel, h)

print("=" * 112)
print(f"CYCLE 31: macro-conditional & path-structure | visible through {VISIBLE_THROUGH}")
print("=" * 112)
results = {}
for name, sig in cands.items():
    m = validate_factor(sig, close_panel, library=lib, fwd_cache=fwd_cache)
    ic, icir = abs(m["ic"]), abs(m["icir"])
    passed = (ic >= 0.007) and (icir >= 0.084) and (m.get("max_abs_library_correlation", 1.0) < 0.5)
    results[name] = {"metrics": m, "pass": passed}
    print(f"[{name}] IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} "
          f"n={m['n_ic_dates']} cov_asset={m['coverage_asset_days']} "
          f"turn={m['turnover_10_rank']} maxlibcorr={m.get('max_abs_library_correlation')} "
          f"decay10={m['decay_ic_by_horizon'].get('10')} decay20={m['decay_ic_by_horizon'].get('20')} "
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
          open("scripts/_miner1_cycle31_explore_results.json", "w"), indent=1, default=float)
print("\nDONE cycle31 exploration")
