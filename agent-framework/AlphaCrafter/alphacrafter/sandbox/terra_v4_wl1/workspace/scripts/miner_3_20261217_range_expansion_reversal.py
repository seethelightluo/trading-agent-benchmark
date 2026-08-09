import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); P[s]=x.set_index('date').sort_index()
# candidate: negative 5-day range expansion (fade unusually wide recent ranges)
R=pd.concat({s:P[s].close.astype(float).pct_change() for s in U},axis=1).sort_index().loc[:'2026-12-16']
TR=pd.concat({s:((P[s].high-P[s].low)/P[s].close).astype(float) for s in U},axis=1).sort_index().reindex(R.index)
F=-(TR.rolling(5,min_periods=4).mean()/TR.rolling(30,min_periods=20).mean()-1)
for k in [1,5,10]:
 y=R.shift(-k).rolling(k).sum().shift(-(k-1)); q=[]; ns=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=pd.Series(q).dropna(); print(k,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'dates',len(q))
print('blocks',pd.Series(q).groupby(pd.cut(np.arange(len(q)),[0,500,850,2000])).mean().to_dict(),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15)
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
F.to_csv('scripts/miner_3_20261217_range_expansion_signal.csv')
