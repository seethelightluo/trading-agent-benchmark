"""miner_2 screen batch (2035-04-25). Fresh candidate factors for the 15-instrument
cross-asset universe under current highvol_elevated_riskoff regime (VIX>60).
Admission gate (benchmark-wide): |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Only uses data visible through 2035-04-24 (previous completed day). Prints n dates/instruments.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_validation_lib import load_panel, load_macro, ic_analysis, print_report, library_corr
import pandas as pd, numpy as np, math

VIS = "2035-04-24"  # previous completed trading day
px = load_panel(max_date=VIS)
ret = px.pct_change()
print("panel shape:", px.shape, "n assets:", px.shape[1], "n dates:", px.shape[0])

vix = load_macro("VIX", max_date=VIS)
dxy = load_macro("DXY", max_date=VIS)

def evalc(f, label):
    res = ic_analysis(f, px, horizon=10, label=label)
    print_report(res)
    ic = res["ic"]; icir = res["icir"]
    gate = (abs(ic) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"  => GATE {'PASS' if gate else 'FAIL'}\n")
    return res, gate

cands = {}

# A. Rank-reversal momentum: cross-sectional momentum rank minus lagged rank (reversal hybrid)
mom10 = px / px.shift(10) - 1
rank_now = mom10.rank(axis=1, pct=True)
mom40 = px / px.shift(40) - 1
rank_lag = mom40.rank(axis=1, pct=True)
cands["rank_reversal_10x40"] = rank_now - rank_lag  # short-term vs long-term rank gap

# B. VIX-regime mean reversion: recent 20d return rescaled by VIX regime (mean-revert when VIX high)
mom5 = px / px.shift(5) - 1
vix_level = vix.reindex(px.index, method='ffill')
# high VIX => favor names that fell recently (mean revert). Negative when VIX high.
cands["vix_gated_mom5_neg"] = -mom5 * (vix_level / vix_level.rolling(60).mean())

# C. US10Y linkage: correlation of asset return with US10Y yield change over 40d (defensive tilt)
us10 = px["US10Y"].pct_change().rolling(40).mean()
# beta of asset on bond yield move; high = bond-proxy (defensive). Use recent 20d
for a in px.columns:
    pass
# compute rolling beta of each asset on US10Y daily change
us10_d = px["US10Y"].pct_change()
asset_beta_bond = ret.rolling(40).cov(us10_d) / us10_d.rolling(40).var()
cands["bond_beta_tilt"] = asset_beta_bond

# D. Equity vs defensive cluster momentum spread (asset-level: own rank within broad vol-momentum)
# momentum vol-adjusted (10d ret / 10d vol) - mean reversion tilt
v20 = ret.rolling(20).std()
cands["vol_adj_mom10"] = (ret.rolling(10).sum()) / v20

# E. Max-drop mean reversion under stress: 10d min return (bearish=negative; want low names to revert)
minr10 = ret.rolling(10).min()
cands["minr10_cs"] = minr10

# F. Trend consistency: fraction of last 20 days positive (capped at 0.5 shift)
pos20 = (ret.rolling(20).apply(lambda x: (x > 0).mean(), raw=True))
cands["consistency20"] = pos20 - pos20.mean(axis=1).shift(1)

# evaluate
for name, f in cands.items():
    try:
        evalc(f, name)
    except Exception as e:
        print(f"[{name}] ERROR {e}")
"