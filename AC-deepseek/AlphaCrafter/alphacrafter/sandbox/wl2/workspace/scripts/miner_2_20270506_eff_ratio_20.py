"""miner_2 2027-05-06 candidate: eff_ratio_20 (Kaufman efficiency ratio).

Idea: trend QUALITY, not direction. An asset whose 20d move is smooth (few
counter-trend days) tends to keep trending; choppy price action reverts.
eff = |close_t - close_{t-20}| / sum(|daily_ret| over 20d). High = smooth trend.
Distinct from raw momentum (orthogonalizes direction vs smoothness).
Direction +1 (smooth trend continuation). Data visible through 2027-05-05.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_20270506_lib import (asset_series, validate_candidate, ASSETS, load_macro,
                                  to_grid, cross_sectional_rank, spearman_ic_matrix,
                                  summarize, fwd_by_horizon_dict, decay_curve,
                                  turnover_10d_rank, coverage_stats, library_pairwise_corr,
                                  GRID, HORIZON, MIN_ASSETS, GATE_IC, GATE_ICIR)

series = asset_series()
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}", flush=True)


def eff_ratio(s, w=20, minp=10):
    close = s["close"]
    path = close.pct_change().abs().rolling(w, min_periods=minp).sum()
    net = (close - close.shift(w)).abs()
    return net / path.replace(0, np.nan)


cand = {s: eff_ratio(df) for s, df in series.items()}
res = validate_candidate("eff_ratio_20", cand, series)
