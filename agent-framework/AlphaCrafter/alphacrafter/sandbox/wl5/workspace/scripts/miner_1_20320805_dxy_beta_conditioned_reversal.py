import os, sys
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# DXY beta-conditioned 10d reversal: reversal magnitude discounted for high absolute DXY beta
xs={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); xs[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(xs).sort_index().ffill()
r=p.pct_change()
macro=pd.read_csv('../persistent/index_data/DXY.csv')
macro['date']=pd.to_datetime(macro['date'])
mc=macro.set_index('date')['close'].astype(float).sort_index().reindex(p.index).ffill().pct_change()
# causal signals at t, forward return t+1..t+10
rows=[]
for i in range(100,len(p)-10):
    dt=p.index[i]
    rr=r.iloc[:i+1]
    # beta over prior 60 completed observations including t; all information through t
    m=mc.iloc[:i+1].iloc[-60:]
    window=rr.iloc[-60:].copy(); valid=m.notna()
    m=m[valid]; window=window.loc[m.index]
    var=m.var()
    beta=window.apply(lambda x: x.cov(m)/var if var>1e-12 else np.nan)
    vol=rr.iloc[-20:].std()*np.sqrt(252)
    rev=-p.iloc[i]/p.iloc[i-10]-1
    sig=rev/(vol.replace(0,np.nan))
    # absolute macro beta penalty, cross-sectional scaled; retain directional reversal
    ab=beta.abs(); med=ab.median(); scale=ab.std()
    fac=sig*(1-(ab-med)/(scale+1e-9)*0.25)
    fwd=p.iloc[i+10]/p.iloc[i]-1
    z=pd.DataFrame({'f':fac,'y':fwd}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)>=8:
        rows.append((dt,z.f.corr(z.y),len(z),z.f.rank(pct=True)))
ics=np.array([x[1] for x in rows])
print('dates',len(rows),'mean_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15)
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2032-08-05')]:
 q=[x[1] for x in rows if pd.Timestamp(a)<=x[0]<=pd.Timestamp(b)]
 print(a,b,len(q),np.mean(q) if q else np.nan,(np.mean(q)/np.std(q,ddof=1)) if len(q)>1 else np.nan)
# signal artifact latest and full rows
out=[]
for dt,ic,n,ranks in rows:
 for s,v in ranks.items(): out.append({'date':dt,'symbol':s,'signal':float(v),'ic':float(ic)})
pd.DataFrame(out).to_csv('scripts/miner_1_20320805_dxy_beta_conditioned_reversal_signal.csv',index=False)
