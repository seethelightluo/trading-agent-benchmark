import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:'2027-01-22'];P[a]=d
p=pd.DataFrame(P); r=p.pct_change(); raw=r.rolling(20,min_periods=20).sum(); sig=raw.sub(raw.mean(axis=1),axis=0).shift(1)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt].rename('s'),r.shift(-1).loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append(spearmanr(z.s,z.y).statistic)
q=pd.Series(rows); print('dates',len(q),'n',len(A),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean(),'coverage',sig.notna().sum(axis=1).mean()/15,'turnover',sig.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20]:
 y=p.shift(-h)/p-1; q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt].rename('s'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.s,z.y).statistic)
 q=pd.Series(q);print('decay',h,q.mean(),len(q))
sig.stack().rename('signal').to_csv('scripts/miner_3_20270122_residual_momentum_signal.csv')
