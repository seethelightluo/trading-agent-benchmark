"""miner_3 2026-07-30: Candidate factor DIST_HIGH_60D (distance from 60d high), v2.

Idea: In cross-asset trend markets, assets far below their recent high are in
drawdown while assets near their high are in uptrends. This measures *position
within the recent range* — distinct from raw momentum.

factor_t = close / rolling_max(close, 60) - 1  (<= 0; more negative = deeper pullback)
Uses min_periods=30 because the panel index is the union of 15 trading
calendars (~27% NaN rows for non-crypto assets).
Direction decided by IC sign (test both + and -).
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
hh = close.rolling(60, min_periods=30).max()
factor = close / hh - 1.0

print("=== DIST_HIGH_60D (v2) ===")
print("factor NaN frac: %.3f" % factor.isna().mean().mean())
fr = forward_returns(close, 10)
for sign, lab in [(1.0, "as-is (near high = high signal)"), (-1.0, "negated (deep pullback = high signal)")]:
    f = factor * sign
    ics = ic_series(f, fr)
    print(f"\n--- direction {lab} ---")
    print("n IC dates:", len(ics))
    m = summary_metrics(ics, f, fr, close, h=10)
    print("metrics:", json.dumps(m, indent=2) if m else "INSUFFICIENT")
    if len(ics):
        print("regime split:", json.dumps(regime_split(ics), indent=2))
        lib_ics = library_ic_series_map(close)
        rho = max_abs_library_corr(ics, lib_ics)
        print("max_abs_library_correlation:", rho)
