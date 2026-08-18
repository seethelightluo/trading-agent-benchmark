"""miner_2 2028-09-07: Bollinger z-score position 20d.

Motivation: (close - SMA20)/std20 measures how far an asset sits above/below
its recent mean in volatility units. Over-extended assets may revert (negative
IC) or continue (positive IC) depending on regime. Distinct from range position
and 52w-high proximity; uses mean/std instead of min/max.
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
sma = close.rolling(W, min_periods=10).mean()
sd = close.rolling(W, min_periods=10).std()
factor = (close - sma) / sd

fwd = fwd_returns(close)
ic10 = rank_ic_series(factor, fwd[10])
print("=== bbz_20 ===")
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
