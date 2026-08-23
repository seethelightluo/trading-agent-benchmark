"""miner_1 factor screening through 2034-09-19.

Explore novel cross-asset candidates not yet persisted in the library
(library is heavy on equity-beta/vol/momentum). Ideas here:
 - dollar & rate sensitivity betas (macro-risk)
 - yield-curve relative value signals between US10Y and CN10Y
 - vol-scaled momentum variants
 - short-term reversal
 - cross-crypto / cross-commodity relative momentum
 - high-low intraday-range signals
Reuses factor_validation_lib on the 15-instrument cross-asset universe.
"""
import sys, json, math
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_validation_lib import (load_panel, load_macro, ic_analysis,
                                   rank_ic_series, align_fwd_returns)

CURRENT = "2034-09-19"
panel = load_panel(max_date=CURRENT)
print("panel", panel.shape, panel.index.min().date(), "->", panel.index.max().date())

vix = load_macro("VIX", CURRENT)
dxy = load_macro("DXY", CURRENT)
dxy = dxy.reindex(panel.index).ffill()

ret = panel.pct_change()
vix_r = vix.reindex(panel.index).pct_change()
dxy_r = dxy.pct_change()
us10_r = panel["US10Y"].pct_change()
cn10_r = panel["CN10Y"].pct_change()
xau_r = ret["XAU"]
btc_r = ret["BTC"]

def beta_series(x, m, win):
    cov = x.rolling(win).cov(m)
    var = m.rolling(win).var()
    return cov / var

def build_beta_panel(cols, m, win):
    return pd.DataFrame({c: beta_series(panel[c], m, win) for c in cols}, index=panel.index)

def neg(f): return -f

cands = {}

# 1) DXY beta neg (dollar-risk defensive)
cands["dxy_beta_60d_neg"] = neg(build_beta_panel(panel.columns, dxy_r, 60))
# 2) US10Y price beta neg (rising-rates defensive)
cands["us10y_beta_60d_neg"] = neg(build_beta_panel(panel.columns, us10_r, 60))
# 3) CN10Y price beta neg
cands["cn10y_beta_60d_neg"] = neg(build_beta_panel(panel.columns, cn10_r, 60))
# 4) vol-scaled 60d momentum (mom/vol)
cands["vol_adj_mom_60x20"] = (panel/panel.shift(60)-1.0)/ret.rolling(20).std()
# 5) short-term reversal 5d
cands["reversal_5d"] = -(panel/panel.shift(5)-1.0)
# 6) XAU beta (inflation-hedge/commodity momentum)
cands["xau_beta_60d"] = build_beta_panel(panel.columns, xau_r, 60)
# 7) BTC-relative 20d mom (risk-on vs crypto)
cands["rel_btc_mom_20d"] = pd.DataFrame({c: ret[c].rolling(20).sum()-btc_r.rolling(20).sum() for c in panel.columns}, index=panel.index)
# 8) intraday range 20d (high-low)/close trend
hl = pd.DataFrame({c: (pd.Series(getattr(panel,c).index,index=panel.index)) for c in panel.columns})  # placeholder
# use close-based range proxy: rolling stdev of daily high-low range via pct range
# skip heavy; use rolling range-to-close
cands["rng_20d"] = None  # build below
# 9) US10Y-CN10Y carry spread level (cross-yield relative value) applied to all assets
cc = (panel["US10Y"]-panel["CN10Y"])
cands["us_cn_spread_20d"] = pd.DataFrame({c: cc.rolling(20).mean() for c in panel.columns}, index=panel.index)
# 10) VIX change 20d neg (risk-level momentum)
cands["vix_down_20d"] = neg(pd.DataFrame({c: vix_r.rolling(20).sum() for c in panel.columns}, index=panel.index))

# range proxy: normalized high-low over last 20d relative to close
cands["rng_20d"] = pd.DataFrame({c: (panel[c]/panel[c].rolling(20).max()-1.0) for c in panel.columns}, index=panel.index)

results = {}
for name, f in cands.items():
    f = f.reindex(panel.index)
    res = ic_analysis(f, panel, horizon=10, label=name)
    res["pass_gate"] = (abs(res["ic"]) >= 0.0070) and (abs(res["icir"]) >= 0.0840)
    results[name] = res
    print("\n=== %s ===" % name)
    print("  ic_signed=%.4f icir=%.4f hit=%.3f n_dates=%d cov=%.3f dge8=%.3f turn=%.3f" %
          (res["ic_signed"], res["icir"], res["ic_hit_ratio"], res["n_ic_dates"],
           res["coverage_asset_days"], res["coverage_dates_ge8"], res["turnover_10d_rank"]))
    print("  decay:", res["decay_ic_by_horizon"])

with open("scripts/miner1_20340920_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("\nDONE")