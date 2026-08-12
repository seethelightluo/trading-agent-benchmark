"""Exploration 3: Realized skewness & downside/upside semideviation asymmetry.

Idea: negative skew / heavy downside semivol signals hedging demand or crash
risk. Cross-sectionally, assets with high downside-vol/upside-vol ratio or
negative skew may carry elevated risk premia -> positive expected returns.
Also test whether this predicts reversal (negative) at short horizons.

Construction (per-asset daily returns r):
- realize_skew_1m: skewness of r over 21d
- realize_skew_3m: skewness over 63d
- downside_ratio_1m / 3m: sqrt(mean(r_-^2))/sqrt(mean(r_+^2)) over window
- dside_ratio_diff: 1m ratio minus 3m ratio (short-term asymmetry change)

Admission horizon 10; gates |IC|>=0.0070, |ICIR|>=0.0840.
"""
import sys, json
sys.path.insert(0, "scripts")
from miner2_20260730_factorlib import load_panel, factor_panel, fwd_ret_panel, validate
import pandas as pd, numpy as np

P = load_panel()
R = P.pct_change()

def skew_win(s, w):
    return s.rolling(w).skew()

def downside_ratio(s, w):
    neg = s.clip(upper=0)
    pos = s.clip(lower=0)
    d = np.sqrt((neg**2).rolling(w).mean())
    u = np.sqrt((pos**2).rolling(w).mean())
    return d / u.replace(0, np.nan)

candidates = [
    ("skew_21", lambda s: skew_win(s, 21), 1),
    ("skew_63", lambda s: skew_win(s, 63), 1),
    ("dside_ratio_21", lambda s: downside_ratio(s, 21), 1),
    ("dside_ratio_63", lambda s: downside_ratio(s, 63), 1),
]

fwd10 = fwd_ret_panel(P, 10)
allf = {}
for label, fn, direction in candidates:
    fvals = factor_panel(R, fn)
    res = validate(fvals, fwd10, label=label, expected_dir=direction)
    print(json.dumps(res))
    allf[label] = fvals

# 1m vs 3m asymmetry change
fvals_diff = allf["dside_ratio_21"] - allf["dside_ratio_63"]
res = validate(fvals_diff, fwd10, label="dside_ratio_21minus63", expected_dir=1)
print(json.dumps(res))

# cross-correlations among new candidates (informational)
print("--- corr among candidates (pooled valid pairs) ---")
fl = pd.concat({k: v.stack() for k, v in allf.items()}, axis=1)
print(fl.corr().round(3).to_string())
