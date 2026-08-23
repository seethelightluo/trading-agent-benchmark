import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change(); x=r.rolling(15,min_periods=10).sum(); common=x.mean(axis=1); vol=r.rolling(45,min_periods=20).std(); sig=-(x.sub(common,axis=0))/vol
print('sig valid rows',sig.notna().sum(axis=1).value_counts().sort_index().tail(), 'all', (sig.notna().sum(axis=1)>=8).sum())
for h in [5,10,20]:
 f=p.shift(-h)/p-1; ics=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8: ics.append(q.iloc[:,0].corr(q.iloc[:,1])); ns.append(len(q))
 q=pd.Series(ics).dropna(); print('h',h,'dates',len(q),'mean_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()); sig.to_csv('scripts/miner_1_20301212_relative_short_reversal_signal.csv')
