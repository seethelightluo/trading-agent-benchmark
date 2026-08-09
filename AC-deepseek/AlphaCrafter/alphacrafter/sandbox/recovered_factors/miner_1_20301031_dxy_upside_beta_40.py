"""Miner 1: validate a DXY-upside beta avoidance signal on the 15-asset universe."""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2030-10-30')

def close(path):
    x=pd.read_csv(path); x['date']=pd.to_datetime(x['date'])
    return x.set_index('date')['close'].astype(float).sort_index().loc[:CUT]
prices=pd.concat({a:close('../persistent/stock_data/'+a+'.csv') for a in ASSETS},axis=1).sort_index()
dxy=close('../persistent/index_data/DXY.csv').reindex(prices.index).ffill()
r=prices.pct_change(); dr=dxy.pct_change()
# Candidate: negative 40-day beta to only positive DXY innovations.  High score =
# relatively resilient when the dollar is strengthening, without using VIX/yield data.
up=dr.clip(lower=0)
# covariance / variance, requiring at least 16 active DXY-up observations
fac=pd.DataFrame(index=r.index,columns=ASSETS,dtype=float)
for i in range(40,len(r)):
    xx=up.iloc[i-40:i]
    if (xx>0).sum()<16 or xx.var()==0: continue
    for a in ASSETS:
        yy=r[a].iloc[i-40:i]
        z=pd.concat([xx,yy],axis=1).dropna()
        if len(z)>=30: fac.loc[fac.index[i],a]=-z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,0].var()
# Cross-sectional median-centre, lag decision signal exactly one completed observation
fac=fac.sub(fac.median(axis=1),axis=0).shift(1)

def evaluate(h):
    fwd=prices.shift(-h)/prices-1; vals=[]; breadth=[]
    for d in fac.index:
        x=fac.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            vals.append((d,spearmanr(x[m],y[m]).statistic)); breadth.append(m.sum())
    z=pd.Series(dict(vals)).dropna()
    return z, {'ic':z.mean(),'icir':z.mean()/z.std(ddof=1) if z.std(ddof=1) else np.nan,
               'hit':(z>0).mean(),'dates':len(z),'mean_n':np.mean(breadth),'min_n':min(breadth)}
print('cutoff',CUT.date(),'price_dates',len(prices),'assets',len(ASSETS))
print('factor_cells',int(fac.notna().sum().sum()),'of',fac.size,'coverage',round(fac.notna().mean().mean(),6))
print('up-DXY active-window dates',int(fac.notna().any(axis=1).sum()))
for h in [1,5,10,20]:
 z,m=evaluate(h); print('H',h,m)
 if h==10:
  for name,start,end in [('2025-2026','2025-01-01','2026-12-31'),('2027-2028','2027-01-01','2028-12-31'),('2029-current','2029-01-01','2030-10-30'),('recent180','2030-05-03','2030-10-30')]:
   q=z.loc[start:end]; print('REGIME',name,'dates',len(q),'ic',round(q.mean(),6),'icir',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,'hit',round((q>0).mean(),4))
# average rank change turnover
rk=fac.rank(axis=1,pct=True); print('rank_change_turnover',round(rk.diff().abs().stack().mean(),6),'cross_section_sd',round(fac.std(axis=1).mean(),6))
print('LIBRARY_NOVELTY: not computed; library JSON records definitions/metrics but no historic signal panels. Per admission contract this is insufficient evidence for persistence.')
