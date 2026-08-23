"""miner_3 (2026-08-20): Sweep O - macro cross-asset correlation dimensions.

Library covers VIX/SOE/CNY betas and VIX beta-conditional. Feels fresh to add
20/40d rolling correlations of each asset's daily return with USDJPY, EURUSD,
DXY return (carry/risk-proxy macro axes). Goal: find dimension with abs IC/ICIR
passing gate and max lib corr < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()


def rt(a):
    return closes[a].pct_change()


def rolling_corr(asset_r, mktr, w, minp=10):
    df = pd.concat([asset_r.rename("a"), mktr.rename("m")], axis=1)
    return df["a"].rolling(w, min_periods=minp).corr(df["m"])


def mr(name):
    return macro[name].reindex(closes["SPX"].index).pct_change()


macro_ret = {k: mr(k) for k in macro}

# Also a normalized 'rate-of-change' style: correlation of asset return with VIX level move (sign-flipped proxy)
candidates = {}
for w in [20, 40]:
    # others: USDJPY (risk-on/carry), EURUSD, DXY
    for opp in ["USDJPY", "EURUSD", "DXY"]:
        candidates[f"corr_{opp.lower()}_{w}"] = {
            a: rolling_corr(rt(a), macro_ret[opp], w) for a in closes
        }

# combination: avg of USDJPY & EURUSD correlations (risk-asset comovement / global liquidity)
combo = {}
for a in closes:
    cj = rolling_corr(rt(a), macro_ret["USDJPY"], 20)
    ce = rolling_corr(rt(a), macro_ret["EURUSD"], 20)
    # distance from zero: how "macro linked" is asset (abs of avg corr)
    combo[a] = (cj + ce) / 2.0
candidates["risklink_avg2020"] = combo

print("assets:", len(closes), "macro:", len(macro))
for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()