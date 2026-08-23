"""miner_1 novel factor screening through 2034-10-24 - robust pass 2.

Handle None metrics safely, add recent-window (last ~2y: 2032-10+) IC to
assess regime-dependent timeliness given the current highvol_elevated_riskoff
regime (VIX 67). Print all candidates with full-sample and recent IC.
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

vix = load_macro("VIX", CURRENT).reindex(panel.index).ffill()
dxy = load_macro("DXY", CURRENT).reindex(panel.index).ffill()
ret = panel.pct_change()
vix_r = vix.pct_change()
dxy_r = dxy.pct_change()
us10_r = panel["US10Y"].pct_change()
cn10_r = panel["CN10Y"].pct_change()
xau_r = ret["XAU"]; btc_r = ret["BTC"]

def beta_series(x, m, win):
    cov = x.rolling(win).cov(m)
    var = m.rolling(win).var()
    return cov / var

def beta_panel(cols, m, win):
    return pd.DataFrame({c: beta_series(panel[c], m, win) for c in cols}, index=panel.index)

cands = {}
cands["dxy_beta_60d_neg"] = -beta_panel(panel.columns, dxy_r, 60)
cands["us10y_trend_60d_neg"] = -pd.DataFrame({c: (panel[c]/panel[c].shift(60)-1.0) for c in panel.columns}, index=panel.index)
cc = panel["US10Y"] - panel["CN10Y"]
cands["us_cn_carry_trend_40d"] = pd.DataFrame({c: cc.rolling(40).mean().pct_change(20) for c in panel.columns}, index=panel.index)
cands["vix_level_trend_20d_neg"] = -pd.DataFrame({c: (vix/vix.shift(20)-1.0) for c in panel.columns}, index=panel.index)
cands["reversal_5d"] = -pd.DataFrame({c: (panel[c]/panel[c].shift(5)-1.0) for c in panel.columns}, index=panel.index)
cands["rel_xau_mom_20d"] = pd.DataFrame({c: ret[c].rolling(20).sum()-xau_r.rolling(20).sum() for c in panel.columns}, index=panel.index)
cands["rel_btc_mom_20d"] = pd.DataFrame({c: ret[c].rolling(20).sum()-btc_r.rolling(20).sum() for c in panel.columns}, index=panel.index)
vol20 = ret.rolling(20).std(); vol60 = ret.rolling(60).std()
cands["vol_accel_20x60"] = pd.DataFrame({c: vol20[c]-vol60[c] for c in panel.columns}, index=panel.index)
cands["dd_depth_60d"] = pd.DataFrame({c: (panel[c]/panel[c].rolling(60).max()-1.0) for c in panel.columns}, index=panel.index)
cands["vol_trend_20x60"] = pd.DataFrame({c: vol20[c]/vol60[c] for c in panel.columns}, index=panel.index)

RECENT = "2032-10-01"
def recent_ic(factor, panel, horizon=10, start=RECENT):
    sub = factor[factor.index >= start]
    psub = panel[panel.index >= start.reindex  if False else panel.index >= start]
    return ic_analysis(sub, psub, horizon=horizon, label="")

results = {}
for name, f in cands.items():
    f = f.reindex(panel.index)
    try:
        res = ic_analysis(f, panel, horizon=10, label=name)
    except Exception as e:
        print(name, "ERROR", e); continue
    # recent window
    sub = f[f.index >= RECENT]; psub = panel[panel.index >= RECENT]
    try:
        rres = ic_analysis(sub, psub, horizon=10, label=name + "_recent")
    except Exception:
        rres = {}
    ic = res.get("ic_signed"); icir = res.get("icir")
    ric = rres.get("ic_signed"); ricir = rres.get("icir")
    pass_gate = (ic is not None and abs(ic) >= 0.0070 and icir is not None and abs(icir) >= 0.0840)
    results[name] = {"full": res, "recent": rres, "pass_gate": pass_gate}
    print("\n=== %s ===" % name)
    print("  FULL : ic_signed=%.4f icir=%.4f hit=%.3f n_dates=%d cov=%.3f dge8=%.3f turn=%.3f pass=%s" %
          (ic or 0.0, icir or 0.0, res.get("ic_hit_ratio") or 0.0, res.get("n_ic_dates") or 0,
           res.get("coverage_asset_days") or 0.0, res.get("coverage_dates_ge8") or 0.0,
           res.get("turnover_10d_rank") or 0.0, pass_gate))
    print("  RECENT: ic_signed=%.4f icir=%.4f n_dates=%d" %
          (ric or 0.0, ricir or 0.0, rres.get("n_ic_dates") or 0))
    print("  decay:", res.get("decay_ic_by_horizon"))

with open("scripts/miner1_20341025_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("\nDONE")