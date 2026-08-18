"""Exploration 5: Trend consistency & drawdown-depth factors (close-only).

Idea: beyond raw momentum (already covered by mom_10d/120d), the *consistency*
of a trend (fraction of positive days) and the *shallow-ness* of recent
drawdowns capture the quality/steadiness of an uptrend. Steady grind-up
assets (high positive-day fraction, shallow drawdowns) may keep outperforming;
deep-drawdown names may be in distress (reversal / mean-reversion).

Uses only close series (all 15 assets fully covered), so no coverage loss.

Construction (per asset daily returns r = pct_change(close), window w):
- pos_frac_w: fraction of days with r>0 over w days        [continuation]
- dd_w: max drawdown over w days (negative); use depth      [reversal]
- updown_emp_prob / directional consistency = |pos_frac-0.5|
Variants tested at w in {20, 60}.

Admission horizon 10; gates |IC|>=0.0070, |ICIR|>=0.0840 (shared 15-name gate).
"""
import sys, json
sys.path.insert(0, "scripts")
from miner2_20260730_factorlib import load_panel, factor_panel, fwd_ret_panel, validate
import pandas as pd, numpy as np

P = load_panel()
R = P.pct_change()
fwd10 = fwd_ret_panel(P, 10)

def pos_frac(s, w):
    return (s > 0).rolling(w).mean()

def max_ddepth(s, w):
    # rolling max drawdown depth, expressed positive (0 = no drawdown)
    cum = (1 + s).cumprod()
    roll_max = cum.rolling(w, min_periods=w).max()
    dd = cum / roll_max - 1.0   # <= 0
    return dd.rolling(w, min_periods=1).min()  # most negative over w

def consistency(s, w):
    pf = (s > 0).rolling(w).mean()
    return (pf - 0.5).abs()

candidates = [
    ("pos_frac_20", lambda s: pos_frac(s, 20), 1),
    ("pos_frac_60", lambda s: pos_frac(s, 60), 1),
    ("max_dd_20", lambda s: max_ddepth(s, 20), -1),   # deep drawdown -> low return
    ("max_dd_60", lambda s: max_ddepth(s, 60), -1),
    ("consistency_20", lambda s: consistency(s, 20), 1),
]

allf = {}
for label, fn, direction in candidates:
    fvals = factor_panel(R, fn)
    res = validate(fvals, fwd10, label=label, expected_dir=direction)
    res["coverage_assets"] = int(fvals.notna().sum(axis=0).gt(0).sum())
    print(json.dumps(res))
    allf[label] = fvals

print("--- corr among new + existing (pooled valid pairs) ---")
import os
existing = {"rng_pos_20d": None, "skew_20d": None, "dside_ratio_21": None}
print("(existing correlation check left to post-Miner gate)")
