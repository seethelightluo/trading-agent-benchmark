"""miner_1 2026-07-30 cycle 24 (part 2): macro-regime conditional factor family.

Motivation: the library is concentrated in momentum/trend/carry. Add factors that
condition per-asset signals on observation-only macro regimes (DXY, USDJPY,
US10Y, VIX). All macro series are OBSERVATION-ONLY; the factor itself is still
per-asset (one value per tradable instrument per date), so cross-sectional IC is
well defined.

Candidates:
  dxy_beta_cond_60x20  : beta(asset_ret, DXY_ret, 60d) x DXY 20d return
  jpy_beta_cond_60x20  : beta(asset_ret, USDJPY_ret, 60d) x USDJPY 20d return
  us10y_beta_cond_60x20: beta(asset_ret, US10Y chg, 60d) x US10Y 20d change
  mom20_skip5_jpy_cond : mom20d(skip5) x sign(USDJPY 20d return)   [risk-on conditional momentum]
  carry12m3m_vix_cond  : carry_12m3m x (1 - VIX 20d return)        [vol-regime conditional carry]
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, macro_series, per_asset, forward_returns,
                         compute_ic, validate_factor, load_library_signals, report)

panel = load_panel()
print(f"panel: {panel.shape}  {panel.index.min().date()}..{panel.index.max().date()}")
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

def rolling_beta(asset_close, macro_ret, window=60, minp=30):
    """Per-asset rolling beta of asset daily return on macro daily return (own calendar)."""
    ar = asset_close.pct_change()
    df = pd.concat([ar.rename("a"), macro_ret.reindex(ar.index).rename("m")], axis=1).dropna()
    cov = df["a"].rolling(window, min_periods=minp).cov(df["m"])
    var = df["m"].rolling(window, min_periods=minp).var()
    return (cov / var).reindex(asset_close.index)

# macro returns
dxy = macro_series("DXY")
jpy = macro_series("USDJPY")
us10y = macro_series("US10Y")
vix = macro_series("VIX")
dxy_ret = dxy.pct_change()
jpy_ret = jpy.pct_change()
us10y_chg = us10y.diff()
vix_ret = vix.pct_change()
dxy_mom20 = dxy / dxy.shift(20) - 1.0
jpy_mom20 = jpy / jpy.shift(20) - 1.0
us10y_chg20 = us10y - us10y.shift(20)
vix_mom20 = vix / vix.shift(20) - 1.0

signals = {}
print("\n=== building per-asset macro-conditional signals ===")

# 1) DXY-beta conditioned on USD trend
beta_dxy = per_asset(panel, rolling_beta, dxy_ret)
signals["dxy_beta_cond_60x20"] = beta_dxy.mul(dxy_mom20.reindex(beta_dxy.index), axis=0)
print("  dxy_beta_cond_60x20:", signals["dxy_beta_cond_60x20"].shape)

# 2) USDJPY-beta conditioned on JPY trend (risk-on proxy)
beta_jpy = per_asset(panel, rolling_beta, jpy_ret)
signals["jpy_beta_cond_60x20"] = beta_jpy.mul(jpy_mom20.reindex(beta_jpy.index), axis=0)
print("  jpy_beta_cond_60x20:", signals["jpy_beta_cond_60x20"].shape)

# 3) US10Y-beta conditioned on yield move
beta_10y = per_asset(panel, rolling_beta, us10y_chg)
signals["us10y_beta_cond_60x20"] = beta_10y.mul(us10y_chg20.reindex(beta_10y.index), axis=0)
print("  us10y_beta_cond_60x20:", signals["us10y_beta_cond_60x20"].shape)

# 4) risk-on conditional momentum: 20d momentum (skip5) x sign(USDJPY 20d)
mom20 = per_asset(panel, lambda s: s.shift(5) / s.shift(25) - 1.0)
jpy_sign = np.sign(jpy_mom20)
signals["mom20_skip5_jpy_cond"] = mom20.mul(jpy_sign.reindex(mom20.index), axis=0)
print("  mom20_skip5_jpy_cond:", signals["mom20_skip5_jpy_cond"].shape)

# 5) vol-regime conditional carry: carry_12m3m x (1 - VIX 20d ret), clipped
carry = per_asset(panel, lambda s: (s.shift(63) / s.shift(252) - 1.0) - (s / s.shift(63) - 1.0))
vix_scale = np.clip(1.0 - vix_mom20.reindex(carry.index), 0.0, 2.0)
signals["carry12m3m_vix_cond"] = carry.mul(vix_scale, axis=0)
print("  carry12m3m_vix_cond:", signals["carry12m3m_vix_cond"].shape)

for fid, sig in signals.items():
    print(f"    {fid}: nan={int(sig.isna().sum().sum())} "
          f"dates_ge8={int((sig.notna().sum(axis=1)>=8).sum())}")

# ---------------------------------------------------------------------------
print("\n=== validation (admission h=10) ===")
library = load_library_signals(panel)
for fid in ["mom20_volproxy60", "mom_curve_volscale", "range_pos_120d",
            "carry_12m3m", "carry_3m1m"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    library[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)

results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=library, fwd_cache=fwd_cache)
    results[fid] = m
    report(fid, m)

passers = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
           and m["n_ic_dates"] >= 800 and m["coverage_dates_ge8"] >= 0.5]
print(f"\nPASSERS (gate + robustness): {passers}")

print("\n=== regime breakdown for passers ===")
regime_out = {}
for fid in passers:
    sig = signals[fid]
    rd = {}
    parts = [fid]
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = (panel.index >= r0) & (panel.index <= r1)
        ic_ser = compute_ic(sig.loc[sub], fwd_cache[str(ADM_H)].loc[sub]).dropna()
        if len(ic_ser) >= 30:
            sd = ic_ser.std()
            icir = ic_ser.mean() / sd if sd > 0 else 0.0
            parts.append(f"{r0[:4]}-{r1[:4]}: {ic_ser.mean():+.4f}/{icir:+.3f}/n={len(ic_ser)}")
            rd[r0[:4]] = {"ic": round(float(ic_ser.mean()), 4),
                          "icir": round(float(icir), 4), "n_dates": int(len(ic_ser))}
    regime_out[fid] = rd
    print("  " + " | ".join(parts))

json.dump({"results": {k: {kk: vv for kk, vv in v.items() if kk != "library_pairwise_corr"}
                       for k, v in results.items()},
           "passers": passers, "regime": regime_out},
          open("scripts/_miner1_cycle24_macro_cond_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle24_macro_cond_results.json")
print("DONE")
