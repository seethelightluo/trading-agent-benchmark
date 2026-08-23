"""miner_2 screen batch (2035-05-23). Vectorized. Fresh candidates for 15-instrument
cross-asset universe under current highvol_elevated_riskoff regime (VIX>60).
Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10. Visible through 2035-05-22.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_validation_lib import load_panel, load_macro, ic_analysis, print_report
import pandas as pd
import numpy as np
import time

VIS = "2035-05-22"
t0 = time.time()
px = load_panel(max_date=VIS)
print("panel:", px.shape, "load_sec", round(time.time() - t0, 2))
ret = px.pct_change()
vix = load_macro("VIX", max_date=VIS)


def evalc(f, label):
    res = ic_analysis(f, px, horizon=10, label=label)
    print_report(res)
    ic = res["ic"]
    icir = res["icir"]
    gate = bool(ic is not None and icir is not None and abs(ic) >= 0.0070 and abs(icir) >= 0.0840)
    print("  => GATE PASS\n" if gate else "  => GATE FAIL\n")
    return res, gate


cands = {}

mom10 = px / px.shift(10) - 1
mom40 = px / px.shift(40) - 1
mom5 = px / px.shift(5) - 1
vix_level = vix.reindex(px.index, method='ffill')
vix_ratio = vix_level / vix_level.rolling(60).mean()

# A. Rank-reversal momentum
cands["rank_reversal_10x40"] = mom10.rank(axis=1, pct=True) - mom40.rank(axis=1, pct=True)

# B. VIX-gated mean reversion (5d)
cands["vix_gated_mom5_neg"] = -mom5 * vix_ratio

# C. US10Y bond beta tilt
us10_d = px["US10Y"].pct_change()
bond_beta = ret.rolling(40).cov(us10_d) / us10_d.rolling(40).var()
cands["bond_beta_tilt"] = bond_beta

# D. Vol-adjusted momentum
cands["vol_adj_mom10"] = ret.rolling(10).sum() / ret.rolling(20).std()

# E. Max-drop mean reversion under stress
cands["minr10_cs"] = ret.rolling(10).min()

# F. Trend consistency (vectorized: fraction of last 20d positive minus its lagged mean)
pos_roll = (ret > 0).rolling(20).mean()
cands["consistency20"] = pos_roll - pos_roll.mean(axis=1).shift(1)

for name, f in cands.items():
    try:
        res, gate = evalc(f, name)
        out = {"res": res, "gate": gate}
    except Exception as e:
        print("[" + name + "] ERROR " + repr(e))