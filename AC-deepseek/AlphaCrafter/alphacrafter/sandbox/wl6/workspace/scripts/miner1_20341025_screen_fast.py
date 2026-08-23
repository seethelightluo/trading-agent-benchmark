"""Fast miner_1 screening through 2034-10-24. Only 10d IC (full + recent), no decay.
Spearman rank IC via pandas corr on daily cross-section (>=8 valid)."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from factor_validation_lib import load_panel, load_macro

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

def fast_ic(f, panel, horizon=10):
    fwd = panel.shift(-horizon)/panel - 1.0
    idx = f.index.intersection(fwd.index)
    ics = []
    for d in idx:
        ff, rr = f.loc[d], fwd.loc[d]
        pair = pd.concat([ff.rename('f'), rr.rename('r')], axis=1).dropna()
        if len(pair) >= 8:
            rho, _ = spearmanr(pair['f'].values, pair['r'].values)
            ics.append(rho)
    ics = np.array(ics)
    if len(ics) == 0:
        return None, None, 0
    return float(ics.mean()), (float(ics.mean())/float(ics.std(ddof=1)) if len(ics) > 1 and ics.std() > 0 else None), len(ics)

RECENT = "2032-10-01"
for name, f in cands.items():
    f = f.reindex(panel.index)
    ic, icir, n = fast_ic(f, panel)
    icr, icrir, nr = fast_ic(f[f.index >= RECENT], panel[panel.index >= RECENT])
    ic, icir = ic or 0.0, icir or 0.0
    icr, icrir = icr or 0.0, icrir or 0.0
    pg = abs(ic) >= 0.0070 and abs(icir) >= 0.0840
    cov = float(f.notna().mean().mean())
    dge8 = float((f.notna().sum(axis=1) >= 8).mean())
    print("== %-24s FULL ic=%+.4f icir=%+.4f nd@%d cov=%.3f dge8=%.3f | RECENT ic=%+.4f icir=%+.4f nd@%d | pass=%s" %
          (name, ic, icir, n, cov, dge8, icr, icrir, nr, pg))
print("DONE")