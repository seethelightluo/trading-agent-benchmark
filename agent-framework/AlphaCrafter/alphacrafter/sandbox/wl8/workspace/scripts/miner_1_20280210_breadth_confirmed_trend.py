import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-02-09')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date)
 P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Lagged cross-sectional trend persistence: 20d return relative to universe median,
# multiplied by a broad-market breadth confirmation (fraction positive over prior 10d).
med=r.median(axis=1); rel=r.rolling(20).sum().sub(med.rolling(20).sum(),axis=0)
bread=(r>0).rolling(10).mean().mean(axis=1)
sig=rel.mul((bread-0.5).abs()+0.5,axis=0).shift(1)
fwd=px.shift(-1)/px-1
v=[]; ds=[]; ns=[]
for d in px.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q):v.append(q);ds.append(d);ns.append(len(g))
a=np.array(v)
print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for lab,fn in {'2020-22':lambda d:d.year<=2022,'2023-25':lambda d:2023<=d.year<=2025,'2026':lambda d:d.year==2026,'2027':lambda d:d.year==2027,'2028':lambda d:d.year>=2028,'recent180':lambda d:d>=END-pd.Timedelta(days=180)}.items():
 z=a[[i for i,d in enumerate(ds) if fn(d)]]
 print(lab,round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,len(z))
