"""Exploration 4: Macro-beta factors.

Idea: each tradable asset's rolling sensitivity (regression beta) to macro
observation series (VIX, DXY, USDJPY, EURUSD, US10Y via index levels)
captures its risk/hedging character. Assets with high positive VIX-beta
(risk assets, loss amplifiers in sell-offs) may earn lower forward returns;
safe/hedge assets (bonds, gold) may behave differently. Test direction via
symmetric |IC| gate.

Construction: for asset returns r_t (daily pct_change) and macro returns m_t
(pct_change of macro level aligned on common dates), beta over rolling window w:
  beta_w = cov(r, m, w) / var(m, w)   (needs >= 12 valid pairs)
Variants: w in {20, 60}; macro in {VIX, DXY, USDJPY, EURUSD}.
US10Y is itself tradable; skip US10Y-beta from macro set interpretation beyond.
"""
import sys, json, os
sys.path.insert(0, "scripts")
from miner2_20260730_factorlib import (load_panel, factor_panel, fwd_ret_panel,
                                       validate, load_close, MACRO, TRADABLE)
import pandas as pd, numpy as np

P = load_panel()
R = P.pct_change()
fwd10 = fwd_ret_panel(P, 10)

MACRO_TICK = {"VIX": "VIX", "DXY": "DXY", "USDJPY": "USDJPY", "EURUSD": "EURUSD"}
macro = {m: load_close(m, macro=True) for m in ["VIX", "DXY", "USDJPY", "EURUSD"]}
macro_r = {m: s.pct_change() for m, s in macro.items()}
# align each macro return series to asset dates
common_dates = R.index

def rolling_beta(asset_r, macro_r, w):
    df = pd.concat([asset_r.rename("a"), macro_r.rename("m")], axis=1)
    df = df.dropna()
    cov = df["a"].rolling(w).cov(df["m"])
    var = df["m"].rolling(w).var()
    beta = cov / var
    return beta.reindex(asset_r.index)

candidates = []
for w in [20, 60]:
    for m, mr in macro_r.items():
        beta = factor_panel(R, lambda s, mr=mr, w=w: rolling_beta(s, mr, w))
        candidates.append((f"beta_{m}_{w}", beta))

for label, fvals in candidates:
    res = validate(fvals, fwd10, label=label, expected_dir=1)
    res["coverage_assets"] = int(fvals.notna().sum(axis=0).gt(0).sum())
    print(json.dumps(res))

# report pooled correlations among macro betas (informational)
print("--- corr among macro betas (pooled) ---")
fl = pd.concat({k: v.stack() for k, v in candidates}, axis=1)
print(fl.corr().round(3).to_string())