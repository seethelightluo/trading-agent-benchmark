import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
frames={}; fut={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].loc[:cut]
 frames[s]=d
 ret=np.log(d/d.shift(1)); cum=ret.rolling(30,min_periods=25).sum()
 def dd(x):
  z=x.cumsum(); return (z-z.cummax()).min()
 dd30=ret.rolling(30,min_periods=25).apply(dd,raw=False).abs(); frames[s]=cum/(dd30+0.01); fut[s]={h:np.log(d.shift(-h)/d) for h in [1,5,10]}
f=pd.concat(frames,axis=1); rows=[]
for dt in f.index:
 for h in [1,5,10]:
  vals=f.loc[dt]; y=pd.Series({s:fut[s][h].get(dt,np.nan) for s in U}); z=pd.concat([vals,y],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=r[r.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1); print(h,'dates',len(q),'avgN',q.n.mean(),'IC',m,'ICIR',m/sd,'hit',(q.ic>0).mean())
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).dropna().mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20261105_trend_resilience_signal.csv',index=False); print('artifact rows',len(out))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15')]:
 q=r[(r.h==1)&(r.date.between(a,b))]; print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
