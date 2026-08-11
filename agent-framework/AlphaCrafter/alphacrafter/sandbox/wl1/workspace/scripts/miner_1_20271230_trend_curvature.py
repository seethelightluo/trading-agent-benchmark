import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None: d=get_index_daily_data(s,days=4000)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Novel candidate: multi-horizon trend curvature. Short trend is rewarded only when it improves on the
# slow trend; normalization by recent volatility makes cross-asset magnitudes comparable. Lagged one day.
curv=(P.pct_change(15)-0.5*P.pct_change(60))/(r.rolling(20).std()+.003)
f=curv.shift(1)
print('assets',len(px),'dates',len(P))
for h in [5,10,20]:
 I=[];Ns=[]; ds=[]
 for i in range(len(P)-h):
  q=pd.concat([f.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic);Ns.append(len(q));ds.append(P.index[i])
 a=np.array(I); ds=pd.DatetimeIndex(ds)
 print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,5),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,st in [('2025+', '2025-01-01'),('2026+','2026-01-01'),('2027','2027-01-01')]:
  z=a[ds>=pd.Timestamp(st)];print(lab,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift()).abs().mean(axis=1).mean(),6))
f.to_csv('scripts/miner_1_20271230_trend_curvature_signal.csv',index_label='date')
