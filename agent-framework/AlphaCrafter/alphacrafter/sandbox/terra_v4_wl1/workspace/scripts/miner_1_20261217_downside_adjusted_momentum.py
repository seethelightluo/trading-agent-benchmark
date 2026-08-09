import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
d={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); d[s]=x.loc[:END,'close'].astype(float)
px=pd.concat(d,axis=1); r=px.pct_change(); ret20=px.pct_change(20)
down=np.sqrt((r.clip(upper=0)**2).rolling(20,min_periods=15).mean())
f=ret20/down
for h in [1,5,10]:
 vals=[]; nms=[]; dates=[]; fr=px.shift(-h)/px-1
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(a)>=8: vals.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); nms.append(len(a)); dates.append(dt)
 z=np.array(vals); ic=np.mean(z); sd=np.std(z,ddof=1); ir=ic/sd*np.sqrt(252)
 print('h',h,'dates',len(z),'avg_names',np.mean(nms),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,np.mean(z>0)))
 if h==1:
  print('period',dates[0],dates[-1])
  for a,b in [(pd.Timestamp('2020-01-01'),pd.Timestamp('2022-12-31')),(pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31')),(pd.Timestamp('2025-01-01'),END)]:
   q=np.array([v for v,dt in zip(z,dates) if a<=dt<=b]); print('regime',a.year,'n',len(q),'ic %.6f ir %.6f'%(np.mean(q),np.mean(q)/np.std(q,ddof=1)*np.sqrt(252)))
ranks=f.rank(axis=1,pct=True); ch=(ranks-ranks.shift(1)).abs().mean(axis=1)
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',ch.mean())
