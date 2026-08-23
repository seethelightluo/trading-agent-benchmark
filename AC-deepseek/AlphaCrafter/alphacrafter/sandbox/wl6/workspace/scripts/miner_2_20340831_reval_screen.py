"""miner_2 2034-08-31: revalidate current ensemble + screen new candidates.
Visible through 2034-08-30. Shared gates: abs(IC)>=0.0070, abs(ICIR)>=0.0840.
ICIR = mean/std of daily cross-sectional Spearman rank IC. Min 8 instruments/date.
Frozen (constant) names excluded from IC observations.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

VISIBLE = "2034-08-30"
TRADABLE = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
            'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['DXY','USDCNY','USDJPY','EURUSD','VIX']

def load(sym, ddir="../persistent/stock_data", cutoff=VISIBLE):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    return df[df["date"] <= pd.Timestamp(cutoff)].set_index("date")["close"].astype(float).sort_index()

px = pd.DataFrame({s: load(s) for s in TRADABLE})
obs = {s: load(s, "../persistent/index_data/") for s in OBS}
ret = px.pct_change()
fwd10 = px.shift(-10)/px - 1
frozen = [s for s in TRADABLE if px[s].nunique() <= 1]
print("panel:", len(px), px.index.min().date(), "->", px.index.max().date(), "frozen:", frozen)

def ic_series(fac, fwdmat=fwd10, min_valid=8):
    dates, ics = [], []
    common = fac.index.intersection(fwdmat.index)
    for dt in common:
        f = fac.loc[dt]; r = fwdmat.loc[dt]
        m = pd.notna(f) & pd.notna(r)
        m = m & ~pd.Series(f.index.isin(frozen), index=f.index)
        if m.sum() < min_valid: continue
        fv = f[m].values.astype(float); rv = r[m].values.astype(float)
        if np.std(fv) < 1e-12 or np.std(rv) < 1e-12: continue
        rho,_ = spearmanr(fv, rv)
        if np.isfinite(rho): dates.append(dt); ics.append(rho)
    return pd.Series(ics, index=dates)

def report(name, fac, full=True, show_recent=True):
    s = ic_series(fac); n=len(s)
    if n==0: print(f"{name}: NO IC DATES"); return None
    mu,sd=s.mean(),s.std(); icir=mu/sd if sd>0 else np.nan; hit=(s>0).mean()
    line=f"{name}: n={n} IC={mu:.4f} ICIR={icir:.4f} hit={hit:.3f}"
    if show_recent:
        recent = s[s.index >= "2032-09-01"]
        rmu = recent.mean() if len(recent) else np.nan
        rsd = recent.std() if len(recent)>2 else np.nan
        ricir = rmu/rsd if rsd and rsd>0 else np.nan
        line+=f" | recent2y IC={rmu:.4f} ICIR={ricir:.4f} n={len(recent)}"
    if full:
        dec={}
        for H in [1,3,5,10,20]:
            fh=px.shift(-H)/px-1; sh=ic_series(fac,fh); dec[H]=round(sh.mean(),4) if len(sh) else np.nan
        line+=f" decay={dec}"
    print(line); return dict(n=n,ic=mu,icir=icir,hit=hit,recent_ic=rmu if show_recent else None)

vix = obs['VIX'].pct_change()
vixvar = vix.rolling(60).var()
beta_vix = ret.rolling(60).cov(vix)/vixvar

spx = px['SPX'].pct_change()
spxvar = spx.rolling(60).var()
beta_spx = ret.rolling(60).cov(spx)/spxvar

beta_chi = ret.rolling(60).cov(ret['000688.SH'])/ret['000688.SH'].rolling(60).var()
beta_cn10y = ret.rolling(60).cov(ret['CN10Y'])/ret['CN10Y'].rolling(60).var()
beta_us10y = ret.rolling(60).cov(ret['US10Y'])/ret['US10Y'].rolling(60).var()
dxy = obs['DXY'].pct_change()
beta_dxy = ret.rolling(60).cov(dxy)/dxy.rolling(60).var()

sign_ewma = ret.ewm(span=60).mean().apply(np.sign)
dv = lambda w: ((ret<0)*ret).rolling(w).std()*np.sqrt(252)

ENS = {
 'beta_vix_60d_neg': -beta_vix,
 'beta_chi_60d': beta_chi,
 'vol_beta_spx_60d': beta_spx,
 'sign_ewma_60d': sign_ewma,
 'mom_10d_skip5': px.pct_change(15),
 'mom_120d_skip5': px.pct_change(125),
 'down_vol_ratio_20x120': dv(20)/dv(120),
 'skew_20d_neg': -ret.rolling(20).skew(),
}
print("\n========== REVALIDATION: current ensemble ==========")
for k,v in ENS.items():
    report(k,v)

print("\n========== NEW CANDIDATES ==========")
NEW = {
 'beta_us10y_60d': beta_us10y,
 'beta_cn10y_60d': beta_cn10y,
 'beta_dxy_60d_neg': -beta_dxy,
 'vol_term_10x60': ret.rolling(10).std()/ret.rolling(60).std(),
 'down_vol_ratio_10x60': dv(10)/dv(60),
 'avg_vol_20d_neg': -ret.rolling(20).std()*np.sqrt(252),
 'upside_downside_ratio_20x60':
     (((ret>0)*ret).rolling(20).std())/(((ret<0)*ret).rolling(20).std()),
 'xau_wti_spread_25d': px['XAU'].pct_change(25)-px['WTI'].pct_change(25),
 'btc_ndx_div_15d': px['BTC'].pct_change(15)-px['NDX'].pct_change(15),
 'rsi_14d_neg': None,
}
# RSI
delta = ret.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain/loss.replace(0,np.nan)
NEW['rsi_28d_neg'] = - (100 - 100/(1+rs))
del NEW['rsi_14d_neg']

for k,v in NEW.items():
    report(k,v)