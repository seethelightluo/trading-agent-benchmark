"""miner_2 detailed evaluation 2026-07-30.

Top candidates from broad screen: full validation metrics (IC/ICIR/hit/
coverage/turnover/decay) plus max-abs correlation vs the old 4-factor
library signals to avoid redundancy.
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import json
from factor_validation_lib import (TRADABLE, MIN_INSTR, load_panel, load_macro,
                                   rank_ic_series, ic_analysis, library_corr)

panel = load_panel(max_date="2026-07-29")
fapi = panel

# Rebuild the old library signal set on the same panel (for redundancy audit)
ret = fapi.pct_change()
old_lib = {}
old_lib['mom_10d_skip5'] = fapi.shift(5) / fapi.shift(15) - 1.0
old_lib['mom_120d_skip5'] = fapi.shift(5) / fapi.shift(125) - 1.0
old_lib['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
vix = load_macro('VIX', max_date="2026-07-29")
vixr = vix.pct_change()
beta = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
old_lib['vix_beta_cond_60x20'] = -beta * (vix / vix.shift(20) - 1.0)

def roll_std(x, w):
    return x.rolling(w, min_periods=max(10, w // 2)).std()
def roll_mean(x, w):
    return x.rolling(w, min_periods=max(10, w // 2)).mean()

C = {}
C['dist_52w_low'] = fapi / fapi.rolling(250, min_periods=125).min() - 1.0
C['range_60d'] = (fapi.rolling(60, min_periods=30).max() - fapi.rolling(60, min_periods=30).min()) / fapi
C['risk_adj_mom_20d_skip5'] = (fapi.shift(5) / fapi.shift(25) - 1.0) / roll_std(fapi.pct_change(), 20)
C['parkinson_20d'] = np.log(fapi.rolling(20, min_periods=10).max()/fapi.rolling(20, min_periods=10).min()) / (2*np.sqrt(2*np.log(2)))
C['range_20d'] = (fapi.rolling(20, min_periods=10).max() - fapi.rolling(20, min_periods=10).min()) / fapi
C['mom_20d_downside'] = (fapi.shift(5)/fapi.shift(25)-1.0) * (fapi.pct_change().clip(upper=0).rolling(20, min_periods=10).std() / roll_std(fapi.pct_change(), 20))
C['trend_x_vol'] = (fapi.shift(5)/fapi.shift(25)-1.0) * (roll_std(fapi.pct_change(), 20) / roll_mean(roll_std(fapi.pct_change(), 20), 120))
C['kurt_60d'] = fapi.pct_change().rolling(60, min_periods=30).kurt()
C['downside_ratio_20_60'] = fapi.pct_change().clip(upper=0).rolling(20, min_periods=10).std() / fapi.pct_change().clip(upper=0).rolling(60, min_periods=30).std()
C['skew_60d'] = fapi.pct_change().rolling(60, min_periods=30).skew()
C['mom_20d_skip3'] = fapi.shift(3) / fapi.shift(23) - 1.0
C['vol_ratio_10_60'] = roll_std(fapi.pct_change(), 10) / roll_std(fapi.pct_change(), 60)
C['mdd_60d'] = fapi / fapi.rolling(60, min_periods=30).max() - 1.0

print("panel dates:", len(fapi), "instruments:", fapi.shape[1])
print("=" * 110)
results = {}
for name, sig in C.items():
    res = ic_analysis(sig, fapi, horizon=10, label=name)
    rho = library_corr(sig, old_lib)
    results[name] = (res, rho)
    print(f"--- {name} ---")
    print(f"  IC={res['ic']:>8.4f} (signed {res['ic_signed']:+.4f})  ICIR={res['icir']:>8.4f}  hit={res['ic_hit_ratio']:.3f}  "
          f"n={res['n_ic_dates']}  turn={res['turnover_10d_rank']}  cov={res['coverage_asset_days']:.3f}")
    print(f"  decay={res['decay_ic_by_horizon']}")
    print(f"  max_abs_lib_corr={rho:.3f}")
    gate_ok = (abs(res['ic']) >= 0.0070) and (abs(res['icir']) >= 0.0840) and (rho < 0.5)
    print(f"  GATE: {'PASS' if gate_ok else 'FAIL'}")
    print()
