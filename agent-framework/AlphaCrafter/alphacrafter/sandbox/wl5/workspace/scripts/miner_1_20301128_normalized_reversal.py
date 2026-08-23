import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change(); v=r.rolling(60).std(); sig=-r.rolling(20).sum()/v
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1]))
 q=pd.Series(z).dropna();print('h',h,'dates',len(q),'n',len(q), 'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
q=pd.DataFrame({'s':sig.rank(axis=1,pct=True).diff().abs().mean(axis=1)});print('turn',q.s.mean());sig.to_csv('scripts/miner_1_20301128_normalized_reversal_signal.csv')
