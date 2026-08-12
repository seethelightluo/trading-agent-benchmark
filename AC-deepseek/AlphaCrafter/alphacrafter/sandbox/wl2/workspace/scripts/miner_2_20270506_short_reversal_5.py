"""miner_2 2027-05-06 candidate: short_reversal_5 (pure 5d reversal, no skip).

Idea: at short horizons in a reversal regime (VIX elevated, momentum leaders
rolling over), cross-asset returns mean-revert. Raw factor = close/close.shift(5)-1
(raw 5d momentum). If reversal dominates, raw IC will be negative (use dir -1).
Validated on raw values; sign interpretation in summary.
Data visible through 2027-05-05.
"""
import sys
sys.path.insert(0, "scripts")
from miner_2_20270506_lib import asset_series, validate_candidate

series = asset_series()
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}", flush=True)

cand = {s: df["close"] / df["close"].shift(5) - 1.0 for s, df in series.items()}
res = validate_candidate("short_reversal_5", cand, series)
