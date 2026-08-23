"""miner_2 2034-06-22: revalidate current ensemble + fresh candidate factors.
Visible through 2034-06-21. 15-instrument cross-asset universe (all tradable).
Modern stock_data has full columns (close present); load via shared lib.
Admission gate: abs(IC10)>=0.0070 and abs(ICIR10)>=0.0840. Min 8 instruments/date.
"""
import pandas as pd, numpy as np, time, sys
sys.path.insert(0, 'scripts')
from factor_validation_lib import load_panel, load_macro, ic_analysis, align_fwd_returns, library_corr, TRADABLE
from scipy.stats import spearmanr

VIS = "2034-06-21"
t0 = time.time()
px = load_panel(VIS)
ret = px.pct_change()
obs = {m: load_macro(m, VIS).reindex(px.index).ffill() for m in ['VIX','DXY','USDCNY','USDJPY','EURUSD']}
frozen = [s for s in px.columns if px[s].nunique() <= 1]
print(f"panel={px.shape} n_assets={px.shape[1]} dates={px.index.min().date()}..{px.index.max().date()} frozen={frozen}", flush=True)

FWDS = {H: px.shift(-H)/px - 1 for H in [1,3,5,10,20]}

def fast_ic(fac, H=10, min_valid=8):
    fwd = FWDS[H]
    common = fac.index.intersection(fwd.index)
    dates, ics = [], []
    fz = set(frozen)
    for dt in common:
        fr = fac.loc[dt]; rr = fwd.loc[dt]
        m = pd.notna(fr) & pd.notna(rr)
        m = m & ~fr.index.isin(fz)
        if m.sum() < min_valid: continue
        fv = fr[m].values.astype(float); rv = rr[m].values.astype(float)
        if np.std(fv) < 1e-12 or np.std(rv) < 1e-12: continue
        rho,_ = spearmanr(fv, rv)
        if np.isfinite(rho): dates.append(dt); ics.append(rho)
    return pd.Series(ics, index=dates)

def report(name, fac, show_recent=True):
    s10 = fast_ic(fac, 10)
    if len(s10) == 0:
        print(f"{name}: NO IC10"); return None
    mu, sd = s10.mean(), s10.std(ddof=1)
    icir = mu/sd if sd > 0 else np.nan
    hit = (s10 > 0).mean()
    dec = []
    for H in [1,3,5,10,20]:
        sh = fast_ic(fac, H); dec.append(round(sh.mean(),4) if len(sh) else np.nan)
    rmu = ricir = np.nan
    if show_recent:
        recent = s10[s10.index >= "2032-09-01"]
        if len(recent):
            rmu = recent.mean(); rsd = recent.std(ddof=1)
            ricir = rmu/rsd if rsd and rsd > 0 else np.nan
    gate = (abs(mu) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"{name:24s} n={len(s10)} IC10={mu:+.4f} ICIR10={icir:+.4f} hit={hit:.3f} "
          f"rec2yIC={rmu:+.4f} recICIR={ricir:+.3f} decay={dec} GATE={'PASS' if gate else 'fail'}", flush=True)
    return dict(n=len(s10), ic=mu, icir=icir, hit=hit, rec_ic=rmu, rec_icir=ricir, decay=dec)

vix = obs['VIX'].pct_change()
vixvar = vix.rolling(60).var()
beta_vix = ret.rolling(60).cov(vix)/vixvar
spx = px['SPX'].pct_change(); spxvar = spx.rolling(60).var()
beta_spx = ret.rolling(60).cov(spx)/spxvar
beta_chi = ret.rolling(60).cov(ret['000688.SH'])/ret['000688.SH'].rolling(60).var()
ur = px['US10Y'].pct_change()
beta_us10y = ret.rolling(60).cov(ur)/ur.rolling(60).var()
dxy = obs['DXY'].pct_change()
beta_dxy = ret.rolling(60).cov(dxy)/dxy.rolling(60).var()
sign_ewma = ret.ewm(span=60).mean().apply(np.sign)
dv = lambda w: ((ret<0)*ret).rolling(w).std()*np.sqrt(252)

ENS = {
 'beta_vix_60d_neg': -beta_vix, 'beta_chi_60d': beta_chi, 'vol_beta_spx_60d': beta_spx,
 'sign_ewma_60d': sign_ewma, 'mom_10d_skip5': px.pct_change(15),
 'mom_120d_skip5': px.pct_change(125), 'down_vol_ratio_20x120': dv(20)/dv(120),
 'skew_20d_neg': -ret.rolling(20).skew(),
}
print("===== REVALIDATE CURRENT ENSEMBLE =====")
ens_res = {}
for k, v in ENS.items():
    ens_res[k] = report(k, v)

print("\n===== NEW CANDIDATES =====")
NEW = {
 'beta_us10y_60d': beta_us10y,
 'beta_dxy_60d_neg': -beta_dxy,
 'vol_term_20x60': ret.rolling(20).std()/ret.rolling(60).std(),
 'down_vol_ratio_10x60': dv(10)/dv(60),
 'mom_40d_skip5': px.pct_change(45),
 'mom_60d_skip5': px.pct_change(65),
 'mom40_voladj': px.pct_change(45)/(ret.rolling(30).std()*np.sqrt(252)),
 'av_vol_20d_neg': -ret.rolling(20).std()*np.sqrt(252),
 'updown_ratio_20x60': (((ret>0)*ret).rolling(20).std())/(((ret<0)*ret).rolling(20).std()),
 'dd_from_60high': px/px.rolling(60).max()-1,
 'corr_us10y_neg': -ret.rolling(60).corr(ur),
 'wti_spread_xau': px['XAU'].pct_change(25)-px['WTI'].pct_change(25),
 'btc_ndx_div_15d': px['BTC'].pct_change(15)-px['NDX'].pct_change(15),
}
# RSI 28d prep
delta = ret.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain/loss.replace(0, np.nan)
NEW['rsi_28d_neg'] = -(100 - 100/(1+rs))

new_res = {}
for k, v in NEW.items():
    new_res[k] = report(k, v)

import json
json.dump({'ens':ens_res,'new':new_res}, open('scripts/miner_2_20340622_screen_results.json','w'), indent=1)
print("\ndone %.1fs" % (time.time()-t0), flush=True)