"""miner_1 2026-07-30 cycle 26 (script A): macro-conditional beta family.

Idea (ONE family): per-asset 60d rolling beta of daily return on a macro driver's
daily change, multiplied by the driver's own 20d momentum (same construction as
the admitted dxy_beta_cond_60x20), applied to drivers NOT yet in the library:
USDJPY (yen carry), USDCNY, EURUSD, WTI (oil cycle), XAU (safe-haven rotation),
COPPER (global growth), US10Y/CN10Y (duration/credit cycle), plus one distinct
variant: VIX-level conditional momentum (mom20 x sign of VIX 60d z-score).

Coverage: close-price only (volume missing for 6/15 assets), full panel.
Gates: |IC|>=0.007, |ICIR|>=0.084 at h=10, n>=800, cov_dates_ge8>=0.5,
max_abs_library_correlation vs persisted signal artifacts < 0.5 (audit metadata;
deterministic gate recomputes from artifacts), regime stability check.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, macro_series, per_asset, forward_returns,
                         compute_ic, validate_factor, report, VISIBLE_THROUGH)

panel = load_panel()
print(f"panel: {panel.shape}  {panel.index.min().date()}..{panel.index.max().date()}")
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---- library: persisted signal artifacts (the real gate input) ----
library = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20", "carry_3m1m", "carry_12m3m",
            "mom_curve_volscale", "range_pos_120d", "vol_surge_20"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    library[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)

# ---- macro driver daily changes ----
def driver_ret(name):
    s = macro_series(name) if name in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"] else panel[name]
    return s.pct_change()

drivers = {
    "usdjpy": driver_ret("USDJPY"), "usdcny": driver_ret("USDCNY"),
    "eurusd": driver_ret("EURUSD"), "wti": driver_ret("WTI"),
    "xau": driver_ret("XAU"), "copper": driver_ret("COPPER"),
    "us10y": driver_ret("US10Y"), "cn10y": driver_ret("CN10Y"),
}

def beta_factory(dret, window=60, minp=30):
    def f(s):
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), dret.reindex(ar.index).rename("m")], axis=1).dropna()
        cov = df["a"].rolling(window, min_periods=minp).cov(df["m"])
        var = df["m"].rolling(window, min_periods=minp).var()
        return (cov / var).reindex(s.index)
    return f

signals = {}
print("\n=== building macro-conditional beta signals ===")
for drv, dret in drivers.items():
    mom20 = (1.0 + dret).rolling(20).apply(np.prod, raw=True) - 1.0  # 20d compounded
    beta_panel = per_asset(panel, beta_factory(dret))
    sig = beta_panel.mul(mom20.reindex(beta_panel.index), axis=0)
    fid = f"beta_cond_{drv}_60x20"
    signals[fid] = sig
    print(f"  {fid}: nan={int(sig.isna().sum().sum())} "
          f"dates_ge8={int((sig.notna().sum(axis=1)>=8).sum())}")

# VIX-level conditional momentum: 20d mom (skip5) x sign(VIX 60d z-score)
vix = macro_series("VIX")
vix_z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
mom20_sig = per_asset(panel, lambda s: s.shift(5) / s.shift(25) - 1.0)
signals["vix_level_mom_cond"] = mom20_sig.mul(np.sign(vix_z).reindex(mom20_sig.index), axis=0)
print(f"  vix_level_mom_cond: nan={int(signals['vix_level_mom_cond'].isna().sum().sum())} "
      f"dates_ge8={int((signals['vix_level_mom_cond'].notna().sum(axis=1)>=8).sum())}")

# ---- validation ----
print("\n=== validation (admission h=10) ===")
results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=library, fwd_cache=fwd_cache)
    results[fid] = m
    report(fid, m)

passers = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
           and m["n_ic_dates"] >= 800 and m["coverage_dates_ge8"] >= 0.5
           and abs(m.get("max_abs_library_correlation", 0.0)) < 0.5]
print(f"\nPASSERS (gate + orthogonality): {passers}")

print("\n=== regime breakdown (10d IC/ICIR) ===")
regime_out = {}
for fid in passers:
    sig = signals[fid]
    parts = [fid]
    rd = {}
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
          open("scripts/_miner1_cycle26_macrobeta_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle26_macrobeta_results.json")
print("DONE")
