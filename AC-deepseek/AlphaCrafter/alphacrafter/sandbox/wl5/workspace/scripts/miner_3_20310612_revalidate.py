# -*- coding: utf-8 -*-
"""miner_3 2031-06-12 revalidation of currently effective library factors (visible through 2031-06-11)."""
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from miner_3_20310612_common import (price_panel, macro_panel, fwd_returns, rank_ic_series,
                                     summarize_ic, decay_analysis, turnover_10d, coverage_stats,
                                     regime_split, lib_trend_r2, lib_semi_down, lib_mom,
                                     lib_dxy_beta, lib_cny_beta, lib_vol_of_vol, lib_tuw,
                                     lib_tail_ratio, lib_vix_beta_cond, lib_kurt, lib_wti_beta,
                                     library_correlation, VISIBLE_THROUGH, WATCH, IC_THRESHOLD, ICIR_THRESHOLD)

close = price_panel("close")
macro = {s: macro_panel(s) for s in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}
fwd10 = fwd_returns(close)[10]

print(f"visible_through={VISIBLE_THROUGH} n_dates={len(close)} n_instruments={close.shape[1]}")

libs = {
    "trend_r2_30_signed": lib_trend_r2(close),
    "semi_down_ratio_20": lib_semi_down(close),
    "mom_120d_skip5": lib_mom(close, 120, 5),
    "dxy_beta_60": lib_dxy_beta(close, macro["DXY"]),
    "cny_beta_60": lib_cny_beta(close, macro["USDCNY"]),
    "vol_of_vol20x60": lib_vol_of_vol(close),
    "mom_10d_skip5": lib_mom(close, 10, 5),
    "time_under_water_120": lib_tuw(close),
    "tail_ratio_20": lib_tail_ratio(close),
    "vix_beta_cond_60x20": lib_vix_beta_cond(close, macro["VIX"]),
    "kurt_20": lib_kurt(close),
    "WTI_BETA_60": lib_wti_beta(close, close["WTI"]),
}

results = {}
for name, panel in libs.items():
    ic_s = rank_ic_series(panel, fwd10)
    summ = summarize_ic(ic_s, label=name)
    decay = decay_analysis(panel, close)
    cov = coverage_stats(panel)
    to = turnover_10d(panel)
    reg = regime_split(ic_s)
    corrs, max_abs = library_correlation(panel, close, macro)
    results[name] = {
        "label": name,
        "horizon": 10,
        "ic": summ["ic"],
        "icir": summ["icir"],
        "ic_hit_ratio": summ["ic_hit_ratio"],
        "n_ic_dates": summ["n_ic_dates"],
        "decay": decay,
        "coverage": cov,
        "turnover_10d": to,
        "regime": reg,
        "max_abs_library_correlation": max_abs,
        "corr_detail": {k: round(v, 3) for k, v in corrs.items() if np.isfinite(v)},
        "gate_pass": bool(abs(summ["ic"]) >= IC_THRESHOLD and abs(summ["icir"]) >= ICIR_THRESHOLD),
    }
    print(f"\n=== {name} ===")
    print(f"  IC={summ['ic']:.4f} ICIR={summ['icir']:.3f} hit={summ['ic_hit_ratio']:.3f} n={summ['n_ic_dates']} "
          f"gate_pass={results[name]['gate_pass']}")
    print(f"  decay={ {k: round(v, 4) for k, v in decay.items()} }")
    print(f"  cov_asset_days={cov['coverage_asset_days']:.3f} cov_ge8={cov['coverage_dates_ge8']:.3f} turnover={to:.3f}")
    print(f"  regime(2031)={reg.get('2031', {})} last90d={reg.get('last90d', {})} last60d={reg.get('last60d', {})}")
    print(f"  max_abs_lib_corr={max_abs:.3f}")

with open("scripts/miner_3_20310612_revalidate_results.json", "w") as f:
    json.dump({"visible_through": VISIBLE_THROUGH, "n_dates": len(close),
               "thresholds": {"ic": IC_THRESHOLD, "icir": ICIR_THRESHOLD},
               "results": results}, f, indent=1, default=str)
print("\nSaved scripts/miner_3_20310612_revalidate_results.json")
