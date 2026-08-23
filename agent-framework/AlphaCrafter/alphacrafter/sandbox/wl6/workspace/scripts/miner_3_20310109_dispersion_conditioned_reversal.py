import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close
P=pd.DataFrame(px).sort_index(); r20=P.pct_change(20); disp=r20.std(axis=1,skipna=True); med=disp.rolling(60,min_periods=30).median(); high=(disp>med).astype(float)
f=(-P.pct_change(5)).mul(1+1.5*high,axis=0)
def rows(h):
 fr=P.shift(-h)/P-1; out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z),high.loc[dt]))
 return pd.DataFrame(out,columns=['date','ic','n','high']).set_index('date')
print('rows',len(P),'instruments',len(P.columns),'span',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 q=rows(h); print('H',h,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit', (q.ic>0).mean())
q=rows(10); print('high_low',q.groupby('high').ic.agg(['count','mean','std'])); print('year',q.groupby(q.index.year).ic.agg(['count','mean']))
print('coverage',q.n.mean()/15,'turnover_proxy',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'high_fraction',high.mean())
