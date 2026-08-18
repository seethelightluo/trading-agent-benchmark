"""miner_2 2028-09-07: Return autocorrelation (lag-1) over 20d.

Motivation: serial correlation of daily returns measures trend persistence /
mean-reversion tendency. Positive autocorrelation -> momentum-like behavior;
negative -> reversal. Not covered by existing library (trend_r2 measures fit
quality, not lag dependence). Hypothesis: assets whose returns show positive
serial dependence tend to keep drifting over the 10d horizon.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from miner_2_20280907_common import (price_panel, macro_panel, fwd_returns,
                                     rank_ic_series, summarize_ic, decay_analysis,
                                     turnover_10d, coverage_stats, regime_split,
                                     library_correlation)

close = price_panel("close")
macro = {m: macro_panel(m) for m in ["DXY", "VIX"]}

W = 20
r = close.pct_change()
factor = r.rolling(W, min_periods=12).apply(
    lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 12 else np.nan, raw=False)

fwd = fwd_returns(close)
ic10 = rank_ic_series(factor, fwd[10])
print("=== ret_autocorr_20 ===")
print(f"factor window: {W} | panel shape: {factor.shape}")
res = summarize_ic(ic10, "10d")
print("decay:", {k: round(v, 4) for k, v in decay_analysis(factor, close).items()})
print("turnover_10d_rank:", round(turnover_10d(factor), 3))
print("coverage:", coverage_stats(factor))
print("regimes:")
regime_split(ic10)
corrs, max_abs = library_correlation(factor, close, macro)
print("max_abs_library_correlation:", round(max_abs, 4))
print("per-lib:", {k: (None if not np.isfinite(v) else round(v, 3)) for k, v in corrs.items()})
