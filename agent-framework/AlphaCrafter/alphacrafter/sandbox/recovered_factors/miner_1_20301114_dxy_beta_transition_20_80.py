"""Miner 1: DXY sensitivity-transition factor (one interpretable macro idea)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2030-11-13')
def getclose(path):
    x=pd.read_csv(path); x['date']=pd.to_datetime(x.date)
    return x.set_index('date').close.astype(float).sort_index().loc[:CUT]
prices=pd.concat({a:getclose('../persistent/stock_data/'+a+'.csv') for a in ASSETS},axis=1).sort_index()
dxy=getclose('../persistent/index_data/DXY.csv').reindex(prices.index).ffill()
r=prices.pct_change(); dx=dxy.pct_change()
# High score: DXY beta has become less positive / more negative in 20 sessions
# versus its 80-session baseline. Betas are ordinary covariance / variance.
fac=pd.DataFrame(index=r.index,columns=ASSETS,dtype=float)
for i in range(80,len(r)):
    for a in ASSETS:
        def beta(n):
            z=pd.concat([dx.iloc[i-n:i],r[a].iloc[i-n:i]],axis=1).dropna()
            return z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,0].var() if len(z)>=int(.75*n) and z.iloc[:,0].var()>0 else np.nan
        bs,bl=beta(20),beta(80)
        fac.iloc[i,fac.columns.get_loc(a)]=bl-bs
fac=fac.sub(fac.median(axis=1),axis=0).shift(1)
def test(h):
    fwd=prices.shift(-h)/prices-1; out=[]; ns=[]
    for d in fac.index:
        m=fac.loc[d].notna()&fwd.loc[d].notna()
        if m.sum()>=8:
            v=spearmanr(fac.loc[d,m],fwd.loc[d,m]).statistic
            if np.isfinite(v): out.append((d,v));ns.append(m.sum())
    z=pd.Series(dict(out)); sd=z.std(ddof=1)
    return z,dict(ic=z.mean(),icir=z.mean()/sd if sd else np.nan,hit=(z>0).mean(),dates=len(z),mean_n=np.mean(ns),min_n=min(ns))
print('candidate dxy_beta_transition_20_80 cutoff',CUT.date(),'price_dates',len(prices),'assets',len(ASSETS))
print('cells',int(fac.notna().sum().sum()),'of',fac.size,'coverage',round(fac.notna().stack().mean(),6),'active_dates',int(fac.notna().any(axis=1).sum()))
for h in (1,5,10,20):
 z,m=test(h);print('H',h,m)
 if h==10:
  for n,s,e in [('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('2029-current','2029-01-01',str(CUT.date())),('recent180','2030-05-17',str(CUT.date()))]:
   q=z.loc[s:e];print('REGIME',n,'dates',len(q),'ic',round(q.mean(),6),'icir',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,'hit',round((q>0).mean(),4))
rk=fac.rank(axis=1,pct=True);print('turnover',round(rk.diff().abs().stack().mean(),6),'mean_cs_sd',round(fac.std(axis=1).mean(),6))
print('NOVELTY: FAILED-EVIDENCE: factor JSON library has no timestamped historical signal panels, so required maximum correlation against all admitted factors cannot be evidenced.')
