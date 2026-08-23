"""miner_2 2035-03-28: revalidate active ensemble snapshot through VISIBLE=2035-03-27.
Gates: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 at horizon 10 (per benchmark admission).
Reports full-period and recent-1y windows, decay, coverage, and turnover proxies.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, time, json
from scipy.stats import spearmanr

VISIBLE = "2035-03-27"
TRADABLE = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
            'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['DXY','USDCNY','USDJPY','EURUSD','VIX']
t0=time.time()
def load(sym, ddir):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df=df.drop_duplicates(subset='date',keep='last')
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
    rec=s[s.index>="2034-03-01"]; rmu=rec.mean() if len(rec) else np.nan
    rsd=rec.std() if len(rec)>2 else np.nan; ricir=rmu/rsd if rsd and rsd>0 else np.nan
    dec=[round(res[h].dropna().mean(),4) if len(res[h].dropna()) else np.nan for h in [1,3,5,10,20]]
    gate = "PASS" if (abs(mu)>=0.0070 and abs(icir)>=0.0840) else "fail"
    print(f"{name}: n={len(s)} IC={mu:.4f} ICIR={icir:.4f} hit={hit:.3f} | 1y IC={rmu:.4f} ICIR={ricir:.4f} | decay={dec} | full[{gate}]", flush=True)
    # turnover proxy: mean abs daily change of cross-sectional rank
    fr = fac.reindex(IDX)
    rank = fr.rank(axis=1)
    turn = rank.diff().abs().mean(axis=1).mean() if len(IDX)>2 else np.nan
    cov = float(fr.notna().mean().mean())
    return dict(ic=mu,icir=icir,hit=hit,n=len(s),rec_ic=rmu,rec_icir=ricir,
                decay=dec, turnover_rank_proxy=float(turn) if np.isfinite(turn) else None,
                coverage=cov)

r = ret
vix = obs['VIX'].pct_change(); vixvar=vix.rolling(60).var()
beta_vix = r.rolling(60).cov(vix)/vixvar
beta_spx = r.rolling(60).cov(r['SPX'])/r['SPX'].rolling(60).var()
beta_chi = r.rolling(60).cov(r['000688.SH'])/r['000688.SH'].rolling(60).var()
sign_ewma = r.ewm(span=60).mean().apply(np.sign)
dv=lambda w: ((r<0)*r).rolling(w).std()*np.sqrt(252)

ENS={
 'beta_vix_60d_neg':-beta_vix,'beta_chi_60d':beta_chi,'vol_beta_spx_60d':beta_spx,
 'sign_ewma_60d':sign_ewma,'mom_10d_skip5':px.pct_change(15),'mom_120d_skip5':px.pct_change(125),
 'down_vol_ratio_20x120':dv(20)/dv(120),'skew_20d_neg':-r.rolling(20).skew(),
}
print("==== ENSEMBLE REVALIDATION 2035-03-28 (horizon 10) ====",flush=True)
resout={}
for k,v in ENS.items():
    o=report(k,v)
    if o: resout[k]=o
resout['regime']=dict(vix_last=float(obs['VIX'].iloc[-1]),
                      vix_60d_avg=float(obs['VIX'][-60:].mean()),
                      vix_30d_avg=float(obs['VIX'][-30:].mean()),
                      spx_20d_ret_pct=float((px['SPX'].iloc[-1]/px['SPX'].iloc[-21]-1)*100),
                      spx_last=float(px['SPX'].iloc[-1]))
json.dump(resout, open('scripts/miner_2_20350328_reval_ensemble.json','w'), indent=1)
print("regime:", json.dumps(resout['regime']))
print("total %.1fs" % (time.time()-t0), flush=True)