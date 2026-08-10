import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); D={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv'); x=pd.read_csv(p); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1,sort=True).sort_index(); w=3
F=-R.rolling(w,min_periods=w).sum()/R.rolling(w,min_periods=w).std()
Y=R.shift(-1); qs=[]; ns=[]; dates=[]
for dt in R.index:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: qs.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); dates.append(dt)
q=pd.Series(qs,index=dates)
print('period',R.index.min(),R.index.max(),'dates',len(q),'avgN',round(np.mean(ns),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),6))
for a,b in [(2019,2022),(2022,2024),(2024,2027)]:
 z=q[(q.index.year>a)&(q.index.year<=b)]; print('REG',a+1,b,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
F.to_csv('scripts/miner_1_20261217_volscaled_reversal3_signal.csv',index_label='date')
