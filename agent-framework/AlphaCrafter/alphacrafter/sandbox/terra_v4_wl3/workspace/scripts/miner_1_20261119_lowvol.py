import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15'); F={};Y={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]; r=np.log(d/d.shift(1)); F[s]=-r.rolling(20,min_periods=16).std(); Y[s]={h:np.log(d.shift(-h)/d) for h in [1,5,10]}
f=pd.concat(F,axis=1); rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.DataFrame({'x':f.loc[dt],'y':{s:Y[s][h].get(dt,np.nan) for s in U}}).dropna()
  if len(z)>=8: rows.append((dt,h,spearmanr(z.x,z.y).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=r[r.h==h]; m=q.ic.mean(); print('H',h,'dates',len(q),'avgN',q.n.mean(),'IC',m,'ICIR',m/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20261119_lowvol_signal.csv',index=False)
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15')]:
 q=r[(r.h==1)&r.date.between(a,b)];print('REG',a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
