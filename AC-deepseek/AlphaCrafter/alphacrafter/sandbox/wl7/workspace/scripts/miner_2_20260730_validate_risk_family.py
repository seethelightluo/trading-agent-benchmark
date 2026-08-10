"""miner_2 final validation: risk family candidates from screen v2.
One idea-family per script; candidates:
  - low_beta_60            : rolling beta vs equal-weight market (60d)
  - rel_mom_20d_skip5      : per-asset 20d momentum (skip 5) minus cross-sectional median
  - max_ret_20d            : max daily return over 20d
  - downside_vol_ratio_20  : downside semi-vol / total vol (20d), flipped to positive
Full metrics: IC/ICIR/hit at horizons 1..20, coverage, turnover, decay, library corr.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_lib import (validate_factor, load_panel, load_macro, per_asset)

panel = load_panel()
macro = load_macro()
rets = panel.pct_change()
mkt = rets.mean(axis=1)

# 1) beta vs EW market
def make_low_beta(win):
    def f(s):
        r = s.pct_change()
        z = pd.concat([r.rename("r"), mkt.reindex(s.index).rename("m")], axis=1)
        return z["r"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var().replace(0, np.nan)
    return per_asset(f)

# 2) relative momentum 20d skip 5
def make_rel_mom(n, skip):
    def f(s):
        return s.shift(skip) / s.shift(n + skip) - 1.0
    def inner(pnl, mcr):
        mom = per_asset(f)(pnl, mcr)
        return mom.sub(mom.median(axis=1), axis=0)
    return inner

# 3) max daily return 20d
def make_max_ret(n):
    return per_asset(lambda s: s.pct_change().rolling(n).max())

# 4) downside vol ratio 20d (flip sign to make IC positive)
def make_downside_vol_ratio(win, flip=False):
    def f(s):
        r = s.pct_change()
        tot = r.rolling(win).std()
        dd = r.clip(upper=0).rolling(win).std()
        v = dd / tot
        return -v if flip else v
    return per_asset(f)

if __name__ == "__main__":
    r1 = validate_factor("low_beta_60", make_low_beta(60))
    r2 = validate_factor("rel_mom_20d_skip5", make_rel_mom(20, 5))
    r3 = validate_factor("max_ret_20d", make_max_ret(20))
    r4 = validate_factor("downside_vol_ratio_20_flip", make_downside_vol_ratio(20, flip=True))
    import json
    summary = {
        "low_beta_60": {k: r1[k] for k in ("ic_h10", "icir_h10", "hit_h10", "n_dates_h10",
                        "coverage_asset_days", "coverage_dates_ge8", "turnover_10d_rank",
                        "max_abs_library_correlation", "admission_gate", "direction")},
        "rel_mom_20d_skip5": {k: r2[k] for k in ("ic_h10", "icir_h10", "hit_h10", "n_dates_h10",
                        "coverage_asset_days", "coverage_dates_ge8", "turnover_10d_rank",
                        "max_abs_library_correlation", "admission_gate", "direction")},
        "max_ret_20d": {k: r3[k] for k in ("ic_h10", "icir_h10", "hit_h10", "n_dates_h10",
                        "coverage_asset_days", "coverage_dates_ge8", "turnover_10d_rank",
                        "max_abs_library_correlation", "admission_gate", "direction")},
        "downside_vol_ratio_20": {k: r4[k] for k in ("ic_h10", "icir_h10", "hit_h10", "n_dates_h10",
                        "coverage_asset_days", "coverage_dates_ge8", "turnover_10d_rank",
                        "max_abs_library_correlation", "admission_gate", "direction")},
    }
    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1, default=str))
