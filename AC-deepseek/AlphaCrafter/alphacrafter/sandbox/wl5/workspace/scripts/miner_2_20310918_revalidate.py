"""miner_2 revalidation of library factors - 2031-09-18 cycle (visible through 2031-09-17)."""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from miner_2_20310918_common import (
    price_panel, macro_panel, fwd_returns, rank_ic_series, summarize_ic,
    decay_analysis, turnover_10d, coverage_stats, regime_split,
    lib_trend_r2, lib_semi_down, lib_mom, lib_beta, lib_vol_of_vol, lib_tuw,
    lib_tail_ratio, lib_vix_beta_cond, lib_kurt, lib_wti_beta,
    VISIBLE_THROUGH, IC_THRESHOLD, ICIR_THRESHOLD,
)

close = price_panel("close")
macro = {s: macro_panel(s) for s in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}
fwd10 = fwd_returns(close, (10,))[10]
print(f"panel shape: {close.shape}, date range {close.index.min().date()} .. {close.index.max().date()}")

factors = {
    "trend_r2_30_signed": lib_trend_r2(close),
    "semi_down_ratio_20": lib_semi_down(close),
    "mom_120d_skip5": lib_mom(close, 120, 5),
    "dxy_beta_60": lib_beta(close, macro["DXY"]),
    "cny_beta_60": lib_beta(close, macro["USDCNY"]),
    "vol_of_vol20x60": lib_vol_of_vol(close),
    "mom_10d_skip5": lib_mom(close, 10, 5),
    "time_under_water_120": lib_tuw(close),
    "tail_ratio_20": lib_tail_ratio(close),
    "vix_beta_cond_60x20": lib_vix_beta_cond(close, macro["VIX"]),
    "kurt_20": lib_kurt(close),
    "WTI_BETA_60": lib_wti_beta(close, close["WTI"]),
}

rows = []
for name, panel in factors.items():
    ic_s = rank_ic_series(panel, fwd10)
    s = summarize_ic(ic_s, name)
    dec = decay_analysis(panel, close)
    to = turnover_10d(panel)
    cov = coverage_stats(panel)
    reg = regime_split(ic_s)
    s.update({"decay_10d": dec.get(10), "decay_1d": dec.get(1), "decay_20d": dec.get(20),
              "turnover_10d": to, "coverage_asset_days": cov["coverage_asset_days"],
              "coverage_dates_ge8": cov["coverage_dates_ge8"], "regime": reg})
    s["gate_pass"] = bool(abs(s["ic"]) >= IC_THRESHOLD and abs(s["icir"]) >= ICIR_THRESHOLD)
    rows.append(s)
    print(f"\n=== {name} ===")
    print(f"  IC(10d)={s['ic']:.4f}  ICIR={s['icir']:.4f}  hit={s['ic_hit_ratio']:.3f}  n={s['n_ic_dates']}  gate={'PASS' if s['gate_pass'] else 'FAIL'}")
    print(f"  decay 1d/3d/5d/10d/20d: {dec.get(1):.4f}/{dec.get(3):.4f}/{dec.get(5):.4f}/{dec.get(10):.4f}/{dec.get(20):.4f}")
    print(f"  turnover_10d={to:.3f}  coverage={cov['coverage_asset_days']:.3f} ge8={cov['coverage_dates_ge8']:.3f}")
    for k, v in reg.items():
        print(f"  regime[{k}]: ic={v['ic']:.4f} icir={v['icir']:.4f} n={v['n']}")

out = {"visible_through": VISIBLE_THROUGH, "factors": rows}
with open("scripts/miner_2_20310918_revalidate_results.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("\nsaved scripts/miner_2_20310918_revalidate_results.json")
