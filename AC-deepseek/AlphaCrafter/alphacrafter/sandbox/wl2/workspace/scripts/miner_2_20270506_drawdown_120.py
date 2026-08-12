"""miner_2 2027-05-06 candidate: drawdown_120 (depth below 120d rolling max).

Idea: contrarian/crash-recovery signal. raw = close/rolling_max(close,120)-1 (<=0).
Deeper drawdown -> more negative raw. If reversal works: fwd return higher for
deeper drawdown -> raw IC negative (dir -1). If momentum works: raw IC positive.
Validated on raw values; sign interpretation in summary.
Data visible through 2027-05-05.
"""
import sys
sys.path.insert(0, "scripts")
from miner_2_20270506_lib import asset_series, validate_candidate

series = asset_series()
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}", flush=True)

cand = {s: df["close"] / df["close"].rolling(120, min_periods=60).max() - 1.0
        for s, df in series.items()}
res = validate_candidate("drawdown_120", cand, series)
