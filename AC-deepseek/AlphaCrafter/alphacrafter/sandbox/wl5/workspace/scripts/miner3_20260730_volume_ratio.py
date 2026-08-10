"""miner_3 2026-07-30: Candidate factor VOL_RATIO_5X60.

Idea: Volume expansion/contraction is a cross-asset participation signal.
When an asset's recent 5d average volume is high relative to its 60d average,
it signals elevated conviction/flow activity, which in liquid cross-asset
markets often precedes continued directional moves (trend confirmation) or
marks climaxes (reversal) depending on regime. Direction decided by IC sign.

factor_t = mean(volume, 5) / mean(volume, 60) - 1
"""
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, library_ic_series_map,
                             max_abs_library_corr, regime_split, load_panel)

close = closes_panel()
frames = load_panel()
vol = pd.DataFrame({s: df.set_index("date")["volume"].astype(float) for s, df in frames.items()}).sort_index()
vol = vol.reindex(close.index)
v5 = vol.rolling(5).mean()
v60 = vol.rolling(60).mean()
factor = v5 / v60 - 1.0

fr = forward_returns(close, 10)
ics = ic_series(factor, fr)
print("=== VOL_RATIO_5X60 ===")
print("n IC dates:", len(ics), "| first:", ics.index.min().date(), "| last:", ics.index.max().date())
m = summary_metrics(ics, factor, fr, close, h=10)
print("metrics:", json.dumps(m, indent=2) if m else "INSUFFICIENT")
print("regime split:", json.dumps(regime_split(ics), indent=2))

lib_ics = library_ic_series_map(close)
rho = max_abs_library_corr(ics, lib_ics)
print("max_abs_library_correlation:", rho)
print("library IC series lengths:", {k: len(v) for k, v in lib_ics.items()})
