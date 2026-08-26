import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-09-11'); P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=end].set_index('date'); P[s]=d.close
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
# Trend consistency: lagged 40-session return multiplied by fraction of positive daily returns.
# Both components are shifted before signal use; final shift prevents same-day information.
ret40=px.shift(5)/px.shift(45)-1
breadth=(r.shift(5)>0).rolling(40,min_periods=25).mean()
sig=(ret40*breadth).shift(1)
allres={}
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; z=[]; ns=[]; ds=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
   if np.isfinite(q): z.append(q);ns.append(len(a));ds.append(dt)
 z=np.array(z); allres[h]=(z,ds)
 print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
z,ds=allres[10]; rank=sig.rank(axis=1,pct=True)
print('TURN',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'COVERAGE',round(sig.notna().sum().sum()/sig.size,4),'assets',len(U))
for lab,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-28','2026-01-01','2028-09-11')]:
 q=z[[a<=str(d.date())<=b for d in ds]]; print('REG',lab,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
sig.index.name='date';sig.to_csv('scripts/miner_2_20280911_trend_consistency_signal.csv')
print('range',px.index.min().date(),px.index.max().date())
