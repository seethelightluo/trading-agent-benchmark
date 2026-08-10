"""miner_2 2026-07-30: Candidate factor DXY_BETA_COND_60X20.

Idea: USD strength is a global risk-off/anti-inflation driver that hits
commodities, EM-linked indices and crypto differently. An asset's beta to the
DXY, interacted with the recent 20d DXY move, should rank assets by their
sensitivity to the prevailing dollar trend.

factor_t = -beta(asset_ret, DXY_ret, 60) * (DXY_t / DXY_{t-20} - 1)
Positive for USD-sensitive (high dollar-beta) assets when the dollar recently
fell, and for USD-hedges when the dollar rose. Direction decided by IC sign.
"""
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, library_ic_series_map,
                             max_abs_library_corr, regime_split)

close = closes_panel()
ret = close.pct_change()
dxy = macro_closes()["DXY"]
dxy_ret = dxy.pct_change()

beta60 = {}
for a in close.columns:
    pair = pd.concat([ret[a].rename("a"), dxy_ret.rename("d")], axis=1).dropna()
    b = pair["a"].rolling(60).cov(pair["d"]) / pair["d"].rolling(60).var()
    beta60[a] = b
bdf = pd.DataFrame(beta60)
factor = -bdf * (dxy / dxy.shift(20) - 1.0)

fr = forward_returns(close, 10)
ics = ic_series(factor, fr)
print("=== DXY_BETA_COND_60X20 ===")
print("n IC dates:", len(ics) if not isinstance(ics, float) else 0)
m = summary_metrics(ics, factor, fr, close, h=10)
print("metrics:", json.dumps(m, indent=2) if m else "INSUFFICIENT")
print("regime split:", json.dumps(regime_split(ics), indent=2))

lib_ics = library_ic_series_map(close)
rho = max_abs_library_corr(ics, lib_ics)
print("max_abs_library_correlation:", rho)
print("library IC series lengths:", {k: len(v) for k, v in lib_ics.items()})
