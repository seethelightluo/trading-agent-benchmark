"""Re-validate currently EFFECTIVE library factors through 2029-05-02 (miner_2).

Admission gate (shared benchmark-wide): abs(IC) >= 0.0070 and abs(ICIR) >= 0.0840
at the 10d horizon on the 15-asset tradable cross-asset universe.
"""
import json
import sys
sys.path.insert(0, "scripts")
from miner_2_20290503_common import (
    price_panel, macro_panel, rank_ic_series, summarize_ic, decay_analysis,
    turnover_10d, coverage_stats, regime_split, lib_trend_r2, lib_semi_down,
    lib_mom, lib_dxy_beta, lib_vol_of_vol, lib_tuw, lib_tail_ratio,
    lib_vix_beta_cond, lib_kurt, lib_wti_beta, IC_THRESHOLD, ICIR_THRESHOLD,
    VISIBLE_THROUGH, CURRENT_DATE,
)

close = price_panel("close")
macro = {s: macro_panel(s) for s in ["DXY", "VIX"]}
fwd10 = close.shift(-10) / close - 1.0

factors = {
    "trend_r2_30_signed": lib_trend_r2(close),
    "semi_down_ratio_20": lib_semi_down(close),
    "mom_120d_skip5": lib_mom(close, 120, 5),
    "dxy_beta_60": lib_dxy_beta(close, macro["DXY"]),
    "vol_of_vol20x60": lib_vol_of_vol(close),
    "mom_10d_skip5": lib_mom(close, 10, 5),
    "time_under_water_120": lib_tuw(close),
    "tail_ratio_20": lib_tail_ratio(close),
    "vix_beta_cond_60x20": lib_vix_beta_cond(close, macro["VIX"]),
    "kurt_20": lib_kurt(close),
    "WTI_BETA_60": lib_wti_beta(close, close["WTI"]),
}

print(f"Re-validation window: through {VISIBLE_THROUGH} | current {CURRENT_DATE}")
print(f"Close panel: {close.shape[0]} dates x {close.shape[1]} assets")
print(f"Gate: |IC|>={IC_THRESHOLD} |ICIR|>={ICIR_THRESHOLD} (10d horizon)\n")

results = {}
for name, panel in factors.items():
    ic_s = rank_ic_series(panel, fwd10)
    m = summarize_ic(ic_s, f"{name:24s}")
    reg = regime_split(ic_s)
    print("   regimes:", {k: f"IC={v['ic']:.3f}/ICIR={v['icir']:.2f}/n={v['n']}" for k, v in reg.items()})
    dec = decay_analysis(panel, close)
    print("   decay:", {k: round(v, 4) for k, v in dec.items()})
    cov = coverage_stats(panel)
    to = turnover_10d(panel)
    print(f"   coverage_asset_days={cov['coverage_asset_days']:.3f} ge8={cov['coverage_dates_ge8']:.3f} turnover10d={to:.3f}")
    passed = abs(m["ic"]) >= IC_THRESHOLD and abs(m["icir"]) >= ICIR_THRESHOLD
    print(f"   >>> {'PASS' if passed else 'FAIL'} gate\n")
    results[name] = {"metrics": m, "regimes": reg, "decay": dec,
                     "coverage": cov, "turnover": to, "passed": passed}

with open("scripts/miner_2_20290503_revalidate_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("Saved scripts/miner_2_20290503_revalidate_results.json")
