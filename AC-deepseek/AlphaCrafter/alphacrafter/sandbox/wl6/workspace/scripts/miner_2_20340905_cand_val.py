"""miner_2 2034-09-05: validate NEW candidate factors on 15-instrument cross-asset universe.
Visible through 2034-09-04. Gates: abs(IC)>=0.0070 (10d), abs(ICIR)>=0.0840.
Full-sample + recent 2y split. Min 8 valid instruments/date.
"""
import pandas as pd, numpy as np, time
from scipy.stats import spearmanr

VISIBLE = "2034-09-04"
TRADABLE = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
            'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['DXY','USDCNY','USDJPY','EURUSD','VIX']

t0=time.time()
def load(sym, ddir="../persistent/stock_data"):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    return df[df["date"] <= pd.Timestamp(VISIBLE)].set_index("date")["close"].astype(float).sort_index()

px = pd.DataFrame({s: load(s) for s in TRADABLE})
obs = {s: load(s, "../persistent/index_data/") for s in OBS}
print("loaded %.1fs" % (time.time()-t0), flush=True)
ret = px.pct_change()
frozen = [s for s in TRADABLE if px[s].nunique() <= 1]
print("panel:", len(px), px.index.min().date(), "->", px.index.max().date(), "frozen:", frozen, flush=True)

FWDS = {H: px.shift(-H)/px - 1 for H in [1,3,5,10,20]}

def ic_series(fac, H=10, min_valid=8):
    fwd = FWDS[H]
    common = fac.index.intersection(fwd.index)
    f = fac.reindex(common); r = fwd.reindex(common)
    dates, ics = [], []
    fzset = set(frozen)
    for dt in common:
        fr = f.loc[dt]; rr = r.loc[dt]
        mask = pd.notna(fr) & pd.notna(rr)
        mask = mask & ~fr.index.isin(fzset)
        if mask.sum() < min_valid: continue
        fv = fr[mask].values.astype(float); rv = rr[mask].values.astype(float)
        if np.std(fv) < 1e-12 or np.std(rv) < 1e-12: continue
        rho,_ = spearmanr(fv, rv)
        if np.isfinite(rho): dates.append(dt); ics.append(rho)
    return pd.Series(ics, index=dates)

def report(name, fac, full=True):
    s = ic_series(fac, 10)
    if len(s)==0:
        print(f"{name}: NO IC 10d", flush=True); return None
    mu=s.mean(); sd=s.std(); icir=mu/sd if sd>0 else np.nan; hit=(s>0).mean()
    recent = s[s.index >= "2032-09-01"]
    rmu = recent.mean() if len(recent) else np.nan
    rsd = recent.std() if len(recent)>2 else np.nan
    ricir = rmu/rsd if rsd and rsd>0 else np.nan
    dec=[]
    for H in [1,3,5,10,20]:
        sh=ic_series(fac,H)
        dec.append(round(sh.mean(),4) if len(sh) else np.nan)
    tr = (fac.diff().abs().mean(axis=1)).dropna()
    to = float(tr.mean()) if len(tr) else np.nan
    cov = float(pd.notna(fac).mean().mean())
    print(f"{name}: n={len(s)} IC10={mu:.4f} ICIR10={icir:.4f} hit={hit:.3f} "
          f"| rec2y IC={rmu:.4f} ICIR={ricir:.4f} | turnover={to:.4f} cov={cov:.2f} | decay={dec}", flush=True)
    return dict(n=len(s),ic=mu,icir=icir,hit=hit,ic_rec=rmu,icir_rec=ricir,turnover=to,cov=cov)

vix = obs['VIX'].pct_change()
vixvar = vix.rolling(60).var()
beta_vix = ret.rolling(60).cov(vix)/vixvar
spx = px['SPX'].pct_change(); spxvar = spx.rolling(60).var()
beta_spx = ret.rolling(60).cov(spx)/spxvar
beta_chi = ret.rolling(60).cov(ret['000688.SH'])/ret['000688.SH'].rolling(60).var()
beta_us10y = ret.rolling(60).cov(ret['US10Y'])/ret['US10Y'].rolling(60).var()
dxy = obs['DXY'].pct_change(); beta_dxy = ret.rolling(60).cov(dxy)/dxy.rolling(60).var()
skew = ret.rolling(20).skew()
rvol20 = ret.rolling(20).std()
rs14 = ret.clip(lower=0).rolling(14).mean()/(-ret.clip(upper=0)).rolling(14).mean()

CAND = {
 'beta_us10y_60d': beta_us10y,
 'beta_dxy_60d_neg': -beta_dxy,
 'vol_term_10x60': ret.rolling(10).std()/ret.rolling(60).std(),
 'skew_x_vol_20d_neg': -(skew * rvol20),
 'skew_chg_20d_neg': -(skew - skew.shift(20)),
 'down_vol_ratio_10x60': (((ret<0)*ret).rolling(10).std()*np.sqrt(252))/(((ret<0)*ret).rolling(60).std()*np.sqrt(252)),
 'xau_copper_rel_20d': px['XAU'].pct_change(20)-px['COPPER'].pct_change(20),
 'btc_ndx_div_15d': px['BTC'].pct_change(15)-px['NDX'].pct_change(15),
 'rsi_14d_neg': -(100-100/(1+rs14)),
}
print("\n==== 2034-09-05 CANDIDATE VALIDATION (visible to %s) ====" % VISIBLE, flush=True)
for k,v in CAND.items(): report(k,v)
print("total %.1fs" % (time.time()-t0), flush=True)