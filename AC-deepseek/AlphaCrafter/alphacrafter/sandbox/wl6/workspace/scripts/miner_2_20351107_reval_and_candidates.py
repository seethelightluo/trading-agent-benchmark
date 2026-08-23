"""miner_2 2035-11-07: revalidate active ensemble snapshot + probe new candidates
through VISIBLE=2035-11-06. Gates: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 at
horizon 10 (benchmark admission for the 15-asset cross-asset universe).
Reports full-period and recent-1y windows, decay, coverage, turnover proxies,
and best-effort max_abs_library_correlation vs recomputable library factors.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, time, json
from scipy.stats import spearmanr, pearsonr

VISIBLE = "2035-11-06"
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
    rec=s[s.index>="2034-11-01"]; rmu=rec.mean() if len(rec) else np.nan
    rsd=rec.std() if len(rec)>2 else np.nan; ricir=rmu/rsd if rsd and rsd>0 else np.nan
    dec=[round(res[h].dropna().mean(),4) if len(res[h].dropna()) else np.nan for h in [1,3,5,10,20]]
    gate = "PASS" if (abs(mu)>=0.0070 and abs(icir)>=0.0840) else "fail"
    print(f"{name}: n={len(s)} IC