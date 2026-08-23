"""miner_1 novel factor screening through 2034-10-24.

Current regime (per ensemble note): highvol_elevated_riskoff, VIX ~67, high
cross-asset dispersion, defensive tilt. Library is heavy on equity-beta/vol/
momentum. Explore NEW orthogonal candidates:
 - dollar (DXY) beta defensive
 - yield curve relative value (US10Y price trend / US10Y-CN10Y spread)
 - VIX level-trend momentum (risk-off persistence)
 - short-term reversal
 - cross-asset relative momentum (BTC- and XAU-relative)
 - vol-of-return acceleration / vol trend regime switch
- Uses factor_validation_lib on the 15-instrument cross-asset universe.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_validation_lib import (load_panel, load_macro, ic_analysis,
                                   rank_ic_series, align_fwd_returns)

CURRENT = "2034-10-24"
panel = load_panel(max_date=CURRENT)
print("panel", panel.shape, panel.index.min().date(), "->", panel.index.max().date())
if panel.shape[0] < 50:
    print("insufficient data"); raise SystemExit

vix = load_macro("VIX", CURRENT).reindex(panel.index).ffill()
dxy = load_macro("DXY", CURRENT).reindex(panel.index).ffill()

ret = panel.pct_change()
vix_r = vix.pct_change()
dxy_r = dxy.pct_change()
us10_r = panel["US10Y"].pct_change()
cn10_r = panel["CN10Y"].pct_change()

def beta_series(x, m, win):
    cov = x.rolling(win).cov(m)
    var = m.rolling(win).var()
    return cov / var

def beta_panel(cols, m, win):
    return pd.DataFrame({c: beta_series(panel[c], m, win) for c in cols}, index=panel.index)

cands = {}

# 1) DXY beta neg (dollar-strength defensive / dollar carry)
cands["dxy_beta_60d_neg"] = -beta_panel(panel.columns, dxy_r, 60)
# 2) US10Y price trend neg (rising-yield defensive for bonds/risky)
cands["us10y_trend_60d_neg"] = -pd.DataFrame({c: (panel[c]/panel[c].shift(60)-1.0)
                                              for c in panel.columns}, index=panel.index)
# 3) US10Y-CN10Y spread trend (yield-curve relative value applied to all assets)
cc = panel["US10Y"] - panel["CN10Y"]
cands["us_cn_carry_trend_40d"] = pd.DataFrame({c: cc.rolling(40).mean().pct_change(20)
                                               for c in panel.columns}, index=panel.index)
# 4) VIX level trend neg (risk-off persistence: fear begets fear)
cands["vix_level_trend_20d_neg"] = -pd.DataFrame({c: (vix/vix.shift(20)-1.0)
                                                  for c in panel.columns}, index=panel.index)
# 5) short-term reversal 5d
cands["reversal_5d"] = -pd.DataFrame({c: (panel[c]/panel[c].shift(5)-1.0)
                                      for c in panel.columns}, index=panel.index)
# 6) XAU-relative momentum 20d (safe-haven relative strength)
xau_r = ret["XAU"]
cands["rel_xau_mom_20d"] = pd.DataFrame({c: ret[c].rolling(20).sum()-xau_r.rolling(20).sum()
                                         for c in panel.columns}, index=panel.index)
# 7) BTC-relative momentum 20d (risk-on relative strength)
btc_r = ret["BTC"]
cands["rel_btc_mom_20d"] = pd.DataFrame({c: ret[c].rolling(20).sum()-btc_r.rolling(20).sum()
                                         for c in panel.columns}, index=panel.index)
# 8) vol acceleration (20d vol minus 60d vol): rising vol -> defensives
vol20 = ret.rolling(20).std(); vol60 = ret.rolling(60).std()
cands["vol_accel_20x60"] = pd.DataFrame({c: vol20[c]-vol60[c] for c in panel.columns},
                                        index=panel.index)
# 9) normalized drawdown depth (distance from 60d high) neg-aware: deep dip = bounce
cands["dd_depth_60d"] = pd.DataFrame({c: (panel[c]/panel[c].rolling(60).max()-1.0)
                                      for c in panel.columns}, index=panel.index)
# 10) low-vs-high vol spread (risk-on normalization): inverse of vol regime
cands["vol_trend_20x60"] = pd.DataFrame({c: vol20[c]/vol60[c] for c in panel.columns},
                                        index=panel.index)

results = {}
for name, f in cands.items():
    f = f.reindex(panel.index)
    res = ic_analysis(f, panel, horizon=10, label=name)
    res["pass_gate"] = (abs(res["ic_signed"] or 0) >= 0.0070) and (abs(res["icir"] or 0) >= 0.0840)
    results[name] = res
    print("\n=== %s ===" % name)
    print("  ic_signed=%.4f icir=%.4f hit=%.3f n_dates=%d cov=%.3f dge8=%.3f turn=%.3f pass=%s" %
          (res["ic_signed"], res["icir"], res["ic_hit_ratio"], res["n_ic_dates"],
           res["coverage_asset_days"], res["coverage_dates_ge8"], res["turnover_10d_rank"], res["pass_gate"]))
    print("  decay:", res["decay_ic_by_horizon"])

with open("scripts/miner1_20341025_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("\nDONE")