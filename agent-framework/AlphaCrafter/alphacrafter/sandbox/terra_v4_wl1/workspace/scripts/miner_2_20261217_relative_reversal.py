import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x[(x.date>=pd.Timestamp('2020-01-01'))&(x.date<=cut)].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index(); r20=R.rolling(20,min_periods=15).sum(); market=r20.mean(axis=1); F=-(r20.sub(market,axis=0))
rows=[]
for dt in R.index:
 z=pd.concat([F.loc[dt].rename('f'),R.shift(-1).loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
for a,b in [('2020-01-01','2021-12-31'),('2022-01-01','2023-12-31'),('2024-01-01','2025-12-31'),('2026-01-01','2026-12-17')]:
 z=q.loc[a:b].ic;print(a,b,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
# rank turnover
rank=F.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
# save artifact for gate provenance
out=F.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20261217_relative_reversal_signal.csv',index=False)
