"""miner_2 revalidation of effective library factors through 2035-11-21."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, json
from miner_2_20351122_common import *
from miner_2_20351122_common import (price_panel, macro_panel, rank_ic_series,
    summarize_ic, decay_analysis, turnover_10d, coverage_stats, regime_split,
    admission_check, CURRENT_DATE, IC_THRESHOLD, ICIR_THRESHOLD)

close = price_panel('close')
macro = {s: macro_panel(s) for s in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}
fwd = {h: close.shift(-h) / close - 1.0 for h in (1, 3, 5, 10, 20)}
fwd10 = fwd[10]

factors = {
    "trend_r2_30_signed": lib_trend_r2(close, 30),
    "semi_down_ratio_20": lib_semi_down(close, 20),
    "mom_120d_skip5": lib_mom(close, 120, 5),
    "dxy_beta_60": lib_beta(close, macro["DXY"], 60),
    "cny_beta_60": lib_beta(close, macro["USDCNY"], 60),
    "vol_of_vol20x60": lib_vol_of_vol(close, 20, 60),
    "mom_10d_skip5": lib_mom(close, 10, 5),
    "time_under_water_120": lib_tuw(close, 120),
    "tail_ratio_20": lib_tail_ratio(close, 20),
    "vix_beta_cond_60x20": lib_vix_beta_cond(close, macro["VIX"], 60, 20),
    "kurt_20": lib_kurt(close, 20),
    "WTI_BETA_60": lib_wti_beta(close, close["WTI"], 60),
}

results = {}
for name, fp in factors.items():
    ic_s = rank_ic_series(fp, fwd10)
    s = summarize_ic(ic_s, name)
    reg = regime_split(ic_s)
    decay = decay_analysis(fp, close, horizons=(1, 3, 5, 10, 20))
    to = turnover_10d(fp)
    cov = coverage_stats(fp)
    passed = admission_check(s["ic"], s["icir"], name)
    results[name] = {
        "factor_id": name,
        "validation_date": CURRENT_DATE,
        "ic_10d": s["ic"], "icir_10d": s["icir"], "ic_hit_10d": s["ic_hit_ratio"],
        "n_ic_dates": s["n_ic_dates"],
        "decay_ic": decay, "turnover_10d": to, **cov,
        "regime": {k: v for k, v in reg.items() if k in ("2031","2032","last180d","last90d","last60d","last30d")},
        "gate_10d": "PASS" if passed else "FAIL",
    }
    print(f"  {name}: IC={s['ic']:.4f} ICIR={s['icir']:.4f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"to10={to:.2f} cov={cov['coverage_asset_days']:.2f} last30d_IC={reg.get('last30d',{}).get('ic',float('nan')):.4f} "
          f"last90d_IC={reg.get('last90d',{}).get('ic',float('nan')):.4f} -> {'PASS' if passed else 'FAIL'}")

with open("scripts/miner_2_20351122_revalidate_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nsaved scripts/miner_2_20351122_revalidate_results.json")