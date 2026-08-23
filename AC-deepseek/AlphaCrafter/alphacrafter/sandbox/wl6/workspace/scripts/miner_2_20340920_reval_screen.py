"""miner_2 2034-09-20: revalidate ensemble + screen new cross-sectional candidates."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, time
from scipy.stats import spearmanr

VISIBLE = "2034-09-19"
TRADABLE = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
            'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['DXY','USDCNY','USDJPY','EURUSD','VIX']
t0=time.time()
def load(sym, ddir):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    return df[df["date"] <= pd.Timestamp(VISIBLE)].set_index("date")["close"].astype(float).sort_index()

px = pd.DataFrame({s: load(s,"../persistent/stock_data") for s in TRADABLE})
obs = {s: load(s,"../persistent/index_data/").reindex(px.index) for s in OBS}
ret = px.pct_change()
FWDS = {H: (px.shift(-H)/px - 1) for H in [1,3,5,10,20]}
IDX = px.index
print("panel:", len(IDX), IDX.min().date(), "->", IDX.max().date(), flush=True)

def ic_all(fac, min_valid=8):
    fac = fac.reindex(IDX)
    out = {H:[] for H in FWDS}
    for dt in IDX:
        fr = fac.loc[dt]
        m0 = pd.notna(fr)
        if int(m0.sum())<min_valid: continue
        for H,fwd in FWDS.items():
            rr = fwd.loc[dt]
            m = m0 & pd.notna(rr)
            if int(m.sum())<min_valid:
                out[H].append((dt,np.nan)); continue
            fv=fr[m].astype(float).values; rv=rr[m].astype(float).values
            if np.std(fv)<1e-12 or np.std(rv)<1e-12:
                out[H].append((dt,np.nan)); continue
            rho,_=spearmanr(fv,rv)
            out[H].append((dt,rho if np.isfinite(rho) else np.nan))
    return {H:pd.Series([x[1] for x in v], index=[x[0] for x in v]) for H,v in out.items()}

def report(name, fac):
    res = ic_all(fac)
    s = res[10].dropna()
    if len(s)==0: print(f"{name}: NO IC"); return None
    mu=s.mean(); sd=s.std(); icir=mu/sd if sd>0 else np.nan; hit=(s>0).mean()
    rec=s[s.index>="2032-09-01"]; rmu=rec.mean() if len(rec) else np.nan
    rsd=rec.std() if len(rec)>2 else np.nan; ricir=rmu/rsd if rsd and rsd>0 else np.nan
    dec=[round(res[h].dropna().mean(),4) if len(res[h].dropna()) else np.nan for h in [1,3,5,10,20]]
    print(f"{name}: n={len(s)} IC={mu:.4f} ICIR={icir:.4f} hit={hit:.3f} | rec2y IC={rmu:.4f} ICIR={ricir:.4f} | decay={dec}", flush=True)
    return dict(ic=mu,icir=icir,hit=hit,n=len(s))

r = ret
vix = obs['VIX'].pct_change(); vixvar=vix.rolling(60).var()
beta_vix = r.rolling(60).cov(vix)/vixvar
beta_spx = r.rolling(60).cov(r['SPX'])/r['SPX'].rolling(60).var()
beta_chi = r.rolling(60).cov(r['000688.SH'])/r['000688.SH'].rolling(60).var()
beta_cn = r.rolling(60).cov(r['CN10Y'])/r['CN10Y'].rolling(60).var()
beta_u10=r.rolling(60).cov(r['US10Y'])/r['US10Y'].rolling(60).var()
dxy=obs['DXY'].pct_change(); beta_dxy=r.rolling(60).cov(dxy)/dxy.rolling(60).var()
sign_ewma = r.ewm(span=60).mean().apply(np.sign)
dv=lambda w: ((r<0)*r).rolling(w).std()*np.sqrt(252)
delta=r.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
rsi28=-(100-100/(1+gain/loss.replace(0,np.nan)))

ENS={
 'beta_vix_60d_neg':-beta_vix,'beta_chi_60d':beta_chi,'vol_beta_spx_60d':beta_spx,
 'sign_ewma_60d':sign_ewma,'mom_10d_skip5':px.pct_change(15),'mom_120d_skip5':px.pct_change(125),
 'down_vol_ratio_20x120':dv(20)/dv(120),'skew_20d_neg':-r.rolling(20).skew(),
}
print("\n==== ENSEMBLE REVALIDATION ====",flush=True)
for k,v in ENS.items(): report(k,v)
NEW={
 'beta_us10y_60d':beta_u10,'beta_cn10y_60d':beta_cn,'beta_dxy_60d_neg':-beta_dxy,
 'vol_term_10x60':r.rolling(10).std()/r.rolling(60).std(),
 'down_vol_ratio_10x60':dv(10)/dv(60),
 'avg_vol_20d_neg':-r.rolling(20).std()*np.sqrt(252),
 'upside_downside_ratio_20x60':(((r>0)*r).rolling(20).std())/(((r<0)*r).rolling(20).std()),
 'rsi_28d_neg':rsi28,
}
print("\n==== NEW CANDIDATES ====",flush=True)
for k,v in NEW.items(): report(k,v)
print("total %.1fs" % (time.time()-t0), flush=True)