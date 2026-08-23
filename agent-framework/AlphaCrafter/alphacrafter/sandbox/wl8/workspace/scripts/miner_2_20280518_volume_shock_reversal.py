import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-05-17')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').sort_index()
# union dates, allowing asynchronous missing observations; cross-section requires >=8 names
idx=sorted(set().union(*[set(v.index) for v in P.values()]))
cl=pd.DataFrame({s:P[s].reindex(idx).close for s in U}); vol=pd.DataFrame({s:P[s].reindex(idx).volume for s in U})
r=cl.pct_change(); shock=vol/vol.rolling(20,min_periods=10).median()-1
sig=(-r*shock.clip(lower=0)).shift(1); fwd=cl.shift(-1)/cl-1
vals=[]; ds=[]; ns=[]
for d in idx:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): vals.append(q);ds.append(d);ns.append(len(g))
a=np.array(vals); print('dates',len(a),'rows',sum(ns),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for lab,fn in {'2020-22':lambda d:d.year<=2022,'2023-25':lambda d:2023<=d.year<=2025,'2026':lambda d:d.year==2026,'2027':lambda d:d.year==2027,'2028':lambda d:d.year>=2028,'recent180':lambda d:d>=END-pd.Timedelta(days=180)}.items():
 z=a[[i for i,d in enumerate(ds) if fn(d)]]; print(lab,'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,'dates',len(z))
