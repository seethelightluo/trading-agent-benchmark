"""Lean miner_1 screening through 2034-10-24. Single-horizon (10d) IC only, no decay,
to keep runtime feasible. Full-sample + recent (2032-10+) windows.
"""
import sys
sys.path.insert(0, 'scripts')
import pandas as pd
from factor_validation_lib import load_panel, load_macro, ic_analysis

CURRENT = "2034-10-24"
panel = load_panel(max_date=CURRENT)
print("panel", panel.shape, panel.index.min().date(), "->", panel.index.max().date())

vix = load_macro("VIX", CURRENT).reindex(panel.index).ffill()
dxy = load_macro("DXY", CURRENT).reindex(panel.index).ffill()
ret = panel.pct_change()
vix_r = vix.pct_change(); dxy_r = dxy.pct_change()
xau_r = ret["XAU"]; btc_r = ret["BTC"]

def beta_series(x, m, win):
    return x.rolling(win).cov(m) / m.rolling(win).var()
def beta_panel(m, win):
    return pd.DataFrame({c: beta_series(panel[c], m, win) for c in panel.columns}, index=panel.index)

cands = {}
cands["dxy_beta_60d_neg"] = -beta_panel(dxy_r, 60)
cands["us10y_trend_60d_neg"] = -pd.DataFrame({c:(panel[c]/panel[c].shift(60)-1) for c in panel.columns}, index=panel.index)
cc = panel["US10Y"] - panel["CN10Y"]
cands["us_cn_carry_trend_40d"] = pd.DataFrame({c: cc.rolling(40).mean().pct_change(20) for c in panel.columns}, index=panel.index)
cands["vix_level_trend_20d_neg"] = -pd.DataFrame({c:(vix/vix.shift(20)-1) for c in panel.columns}, index=panel.index)
cands["reversal_5d"] = -pd.DataFrame({c:(panel[c]/panel[c].shift(5)-1) for c in panel.columns}, index=panel.index)
cands["rel_xau_mom_20d"] = pd.DataFrame({c: ret[c].rolling(20).sum()-xau_r.rolling(20).sum() for c in panel.columns}, index=panel.index)
cands["rel_btc_mom_20d"] = pd.DataFrame({c: ret[c].rolling(20).sum()-btc_r.rolling(20).sum() for c in panel.columns}, index=panel.index)
vol20 = ret.rolling(20).std(); vol60 = ret.rolling(60).std()
cands["vol_accel_20x60"] = pd.DataFrame({c: vol20[c]-vol60[c] for c in panel.columns}, index=panel.index)
cands["dd_depth_60d"] = pd.DataFrame({c:(panel[c]/panel[c].rolling(60).max()-1) for c in panel.columns}, index=panel.index)
cands["vol_trend_20x60"] = pd.DataFrame({c: vol20[c]/vol60[c] for c in panel.columns}, index=panel.index)

RECENT = "2032-10-01"
for name, f in cands.items():
    f = f.reindex(panel.index)
    try:
        res = ic_analysis(f, panel, horizon=10, label=name)
    except Exception as e:
        print(name, "ERR", e); continue
    sub = f[f.index >= RECENT]; psub = panel[panel.index >= RECENT]
    try:
        rres = ic_analysis(sub, psub, horizon=10, label="recent")
    except Exception:
        rres = {}
    ic, icir = res.get("ic_signed"), res.get("icir")
    ric, ricir = rres.get("ic_signed"), rres.get("icir")
    pg = ic is not None and abs(ic) >= 0.0070 and icir is not None and abs(icir) >= 0.0840
    print("== %s ==  FULL ic=%.4f icir=%.4f hit=%.3f nd@%d cov=%.3f dge8=%.3f turn=%.3f | RECENT ic=%.4f icir=%.4f nd@%d | pass=%s" %
          (name, ic or 0.0, icir or 0.0, res.get("ic_hit_ratio") or 0, res.get("n_ic_dates") or 0,
           res.get("coverage_asset_days") or 0, res.get("coverage_dates_ge8") or 0,
           res.get("turnover_10d_rank") or 0, ric or 0.0, ricir or 0.0, rres.get("n_ic_dates") or 0, pg))
print("DONE")