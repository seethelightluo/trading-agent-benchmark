import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; z=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None:d=get_index_daily_data(s,4000)
 if d is None:continue
 d=d.sort_values('date'); r=d.close.pct_change()
 # medium-term trend, skip last 5 sessions; lagged one day
 f=(d.close.shift(6)/d.close.shift(26)-1); fr=d.close.shift(-1)/d.close-1
 z.append(pd.DataFrame({'date':pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'),'f':f,'fr':fr,'symbol':s}))
x=pd.concat(z); A=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.f.nunique()>1:A.append(g.f.corr(g.fr));ns.append(len(g))
a=pd.Series(A).dropna();print('dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'coverage',np.mean(np.array(ns)/15));print('recent',a.tail(500).mean(),'early',a.head(500).mean())
for h in [3,5,10]:
 b=[]
 for dt,g in x.assign(fr2=lambda q: q.f).groupby('date'): pass
 # compute aggregate using shifted close proxy per symbol
 zz=[]
 for s,g in x.groupby('symbol'):
  d=get_stock_daily_data(s,4000)
  if d is None:d=get_index_daily_data(s,4000)
  d=d.sort_values('date'); fr=d.close.shift(-h)/d.close-1; zz.append(pd.DataFrame({'date':pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'),'f':d.close.shift(6)/d.close.shift(26)-1,'fr':fr}))
 q=pd.concat(zz)
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8 and g.f.nunique()>1:b.append(g.f.corr(g.fr))
 print('h',h,'IC',pd.Series(b).dropna().mean(),'n',len(b))
x[['date','symbol','f']].dropna().to_csv('scripts/miner_1_20271118_trend_signal.csv',index=False)
