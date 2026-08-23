"""miner_1 screen new cross-asset factor ideas through 2034-09-11.

Explore candidates not currently in the persisted library (which is heavy on
equity-beta/vol/momentum): dollar-beta, rate-sensitivity-beta/relative value,
vol-scaled momentum, cross-market own-asset momentum alternatives.
Reuses factor_validation_lib on the 15-instrument cross-asset universe.
"""
import sys, math, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_validation_lib import (load_panel, load_macro, ic_analysis,
                                   align_fwd_returns, rank_ic_series)

CURRENT = "2034-09-11"
panel = load_panel(max_date=CURRENT)
print("panel", panel.shape, panel.index.min().date(), "->", panel.index.max().date())

# macro observables
vix = load_macro("VIX", CURRENT)
dxy = load_macro("DXY", CURRENT)
us10y = load_macro("US10Y", CURRENT) if False else None
# US10Y is in tradable panel already; CN10Y too.
us10y_px = panel["US10Y"]
cn10y_px = panel["CN10Y"]

# returns
ret = panel.pct_change()

# ---------- Factor builders ----------
def build_neg_beta_macro(panel_r, macro_ret, win=60):
    """-beta to a macro daily-return series."""
    macro_v = macro_ret.reindex(panel_r.index)
    f = {}
    for col in panel_r.columns:
        x = panel_r[col]
        m = macro_v
        cov = x.rolling(win).cov(m)
        var = m.rolling(win).var()
        beta = cov / var
        f[col] = -beta
    return pd.DataFrame(f, index=panel_r.index)

def build_vol_scaled_mom(panel, win=60, hld=20):
    mom = panel / panel.shift(win) - 1.0
    vol = panel.pct_change().rolling(hld).std()
    return mom / vol

def build_reversal(panel, win=5):
    return -(panel / panel.shift(win) - 1.0)

def build_yield_sens(panel_r, us10_r, win=60):
    """beta to US10Y price return (bonds-like proxy use negative)."""
    f = {}
    for col in panel_r.columns:
        cov = panel_r[col].rolling(win).cov(us10_r)
        var = us10_r.rolling(win).var()
        f[col] = cov / var
    return pd.DataFrame(f, index=panel_r.index)

cands = {}

# 1) DXY beta neg (dollar-risk defensive)
dxy_ret = dxy.pct_change()
cands["dxy_beta_60d_neg"] = build_neg_beta_macro(ret, dxy_ret, 60)

# 2) CPI/rate: beta to US10Y price return, neg (rising-rate defensive = negative bond beta)
us10_r = us10y_px.pct_change()
y = build_yield_sens(ret, us10_r, 60)
cands["us10y_beta_60d_neg"] = -y

# 3) CN10Y beta neg
cn10_r = cn10y_px.pct_change()
c2 = build_yield_sens(ret, cn10_r, 60)
cands["cn10y_beta_60d_neg"] = -c2

# 4) vol-scaled momentum 60/20
cands["vol_adj_mom_60x20"] = build_vol_scaled_mom(panel, 60, 20)

# 5) short-term reversal 5d
cands["reversal_5d"] = build_reversal(panel, 5)

# 6) gold-copper / commodity beta: beta to XAU returns (inflation hedge momentum)
xau_r = ret["XAU"]
g = {}
for col in ret.columns:
    cov = ret[col].rolling(60).cov(xau_r)
    var = xau_r.rolling(60).var()
    g[col] = cov / var
cands["xau_beta_60d"] = pd.DataFrame(g, index=ret.index)

# 7) skew_20d as raw put-skew proxy? already have skew_20d_neg. skip.

# 8) cross-asset relative momentum: momentum of BTC vs equity (risk-on proxy factor)
btc_r = ret["BTC"]
b = {}
for col in ret.columns:
    b[col] = (ret[col]).rolling(20).sum() - (btc_r).rolling(20).sum()
cands["rel_btc_mom_20d"] = pd.DataFrame(b, index=ret.index)

results = {}
for name, f in cands.items():
    f = f.reindex(panel.index)
    res = ic_analysis(f, panel, horizon=10, label=name)
    res["name"] = name
    res["pass_gate"] = (abs(res["ic"]) >= 0.0070) and (abs(res["icir"]) >= 0.0840)
    results[name] = res
    print("\n=== %s ===" % name)
    print("  ic_signed=%.4f icir=%.4f hit=%.3f n_dates=%d" %
          (res["ic_signed"], res["icir"], res["ic_hit_ratio"], res["n_ic_dates"]))
    print("  cov_asset_days=%.3f cov_dates_ge8=%.3f turn=%.3f ic_std=%.4f" %
          (res["coverage_asset_days"], res["coverage_dates_ge8"],
           res["turnover_10d_rank"], res["ic_std"]))
    print("  decay:", res["decay_ic_by_horizon"], "| ic(signed at 10)=", res["ic_signed"])

with open("scripts/miner1_20340911_screen_results.json", "w") as fh:
    json.dump({k: v for k, v in results.items()}, fh, indent=1, default=str)
print("\nDONE")