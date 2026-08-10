"""miner_3 2026-07-30: Candidate factor VOL_RATIO_5X60 (volume expansion), v2.

Idea: Volume expansion/contraction is a cross-asset participation signal.
When an asset's recent 5d average volume is high relative to its 60d average,
it signals elevated conviction/flow activity. Direction decided by IC sign.

factor_t = mean(volume, 5) / mean(volume, 60) - 1
Zero-volume days treated as missing (NaN) so sparse assets are handled fairly;
rolling means use min_periods to keep partial-window coverage.
"""
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split, load_panel,
                             library_ic_series_map, max_abs_library_corr)

close = closes_panel()
frames = load_panel()
vol = pd.DataFrame({s: df.set_index("date")["volume"].astype(float) for s, df in frames.items()}).sort_index()
vol = vol.reindex(close.index)
vol = vol.replace(0.0, np.nan)
v5 = vol.rolling(5, min_periods=3).mean()
v60 = vol.rolling(60, min_periods=30).mean()
factor = v5 / v60 - 1.0

print("=== VOL_RATIO_5X60 (v2) ===")
print("factor NaN frac: %.3f" % factor.isna().mean().mean())
print("assets with >10% valid factor days:", [c for c in factor.columns if factor[c].notna().mean() > 0.1])
fr = forward_returns(close, 10)
ics = ic_series(factor, fr)
print("n IC dates:", len(ics), "| range:", ics.index.min().date() if len(ics) else "NA",
      "->", ics.index.max().date() if len(ics) else "NA")
m = summary_metrics(ics, factor, fr, close, h=10)
print("metrics:", json.dumps(m, indent=2) if m else "INSUFFICIENT")
if len(ics):
    print("regime split:", json.dumps(regime_split(ics), indent=2))
lib_ics = library_ic_series_map(close)
rho = max_abs_library_corr(ics, lib_ics)
print("max_abs_library_correlation:", rho)
