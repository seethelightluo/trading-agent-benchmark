import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-05-03')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').sort_index()
idx=sorted(set.intersection(*[set(v.index) for v in P.values()]))
cl=pd.DataFrame({s:P[s].reindex(idx).close for s in U}); r=cl.pct_change()
disp=r.std(axis=1).rolling(20).mean().shift(1); threshold=disp.rolling(60,min_periods=30).median()
sig=(-cl.pct_change(3).shift(1)).where(disp>threshold, other=np.nan)
fwd=cl.shift(-1)/cl-1
rows=[]; dates=[]; ns=[]
for d in idx:
 g=pd.DataFrame({'signal':sig.loc[d],'fwd':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1:
  q=spearmanr(g.signal,g.fwd).statistic
  if np.isfinite(q): rows.append(q); dates.append(d); ns.append(len(g))
a=np.array(rows); print('dates',len(a),'rows',int(sum(ns)),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for lab,fn in {'2020-22':lambda d:d.year<=2022,'2023-25':lambda d:2023<=d.year<=2025,'2026':lambda d:d.year==2026,'2027':lambda d:d.year==2027,'2028':lambda d:d.year>=2028,'recent180':lambda d:d>=END-pd.Timedelta(days=180)}.items():
 z=a[[i for i,d in enumerate(dates) if fn(d)]]; print(lab,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
sig.to_csv('scripts/miner_1_20280504_dispersion_reversal_signal.csv')
