import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-06-14')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').sort_index()
idx=sorted(set.intersection(*[set(v.index) for v in P.values()])); cl=pd.DataFrame({s:P[s].reindex(idx).close for s in U}); r=cl.pct_change()
# Relative 20d trend: demean lagged 20d returns cross-sectionally, with mild 5d reversal overlay.
m20=r.rolling(20).sum().shift(1); sig=m20.sub(m20.median(axis=1),axis=0)-0.25*r.rolling(5).sum().shift(1)
fwd=cl.shift(-1)/cl-1; vals=[];ds=[];ns=[]
for date in idx:
 g=pd.DataFrame({'s':sig.loc[date],'f':fwd.loc[date]}).dropna()
 if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): vals.append(q);ds.append(date);ns.append(len(g))
a=np.asarray(vals); print('idea=peer_relative_momentum_reversal'); print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for lab,fn in {'2020-22':lambda z:z.year<=2022,'2023-25':lambda z:2023<=z.year<=2025,'2026':lambda z:z.year==2026,'2027':lambda z:z.year==2027,'2028':lambda z:z.year>=2028,'recent180':lambda z:z>=END-pd.Timedelta(days=180)}.items():
 z=a[[i for i,x in enumerate(ds) if fn(x)]]; print(lab,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
