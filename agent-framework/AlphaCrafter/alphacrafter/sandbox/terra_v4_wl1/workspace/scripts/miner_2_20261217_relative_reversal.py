import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index();
# Relative reversal: prior 5d return relative to contemporaneous cross-asset median, inverted.
rel=R.rolling(5,min_periods=5).sum().sub(R.rolling(5,min_periods=5).sum().median(axis=1),axis=0)
F=-rel
for h in [1,5,10]:
 Y=R.shift(-1).rolling(h,min_periods=h).sum().shift(-(h-1)); vals=[]; ns=[]; ds=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=pd.Series(vals,index=ds);print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7),'hit',round((q>0).mean(),4),'coverage',round(len(q)/len(R),4))
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
  z=q.loc[a:b];print(a+'-'+b,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
rank=F.rank(axis=1,pct=True);print('turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'valid',round(F.notna().mean().mean(),6))
F.index.name='date';F.reset_index().to_csv('scripts/miner_2_20261217_relative_reversal_signal.csv',index=False)
