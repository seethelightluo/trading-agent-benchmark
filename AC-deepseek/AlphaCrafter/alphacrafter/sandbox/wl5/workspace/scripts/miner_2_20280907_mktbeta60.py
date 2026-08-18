"""miner_2 2028-09-07: Cross-sectional market beta 60d.

Motivation: beta of each asset's returns to the equal-weight cross-section
return over 60d. Captures risk-on/risk-off exposure; high-beta assets may be
rewarded (or punished) in different regimes. Distinct from dxy_beta_60 (dollar
sensitivity) and evicted single-asset betas (btc_beta/eth_beta/wti_beta).
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

W = 60
r = close.pct_change()
mkt = r.mean(axis=1)  # equal-weight cross-section return
cov = r.rolling(W, min_periods=30).cov(mkt)
var = mkt.rolling(W, min_periods=30).var()
factor = cov / var

fwd = fwd_returns(close)
ic10 = rank_ic_series(factor, fwd[10])
print("=== mkt_beta_60 ===")
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
