import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None:d=get_index_daily_data(s,4000)
 if d is None:continue
 d=d.sort_values('date').copy(); d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'); r=d.close.pct_change()
 # lagged 3d return, cross-sectional demeaned; contrarian signal
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r3':d.close/d.close.shift(3)-1,'fr':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows); x['f']=-(x['r3']-x.groupby('date').r3.transform('mean'))
x[['date','symbol','f']].dropna().to_csv('scripts/miner_1_20271202_relative_reversal_signal.csv',index=False)
for h in [1,3,5,10]:
 x['frh']=x.groupby('symbol').close if False else np.nan
 # reload forward horizons from source per symbol
 ys=[]
 for s,g in x.groupby('symbol'):
  # use known r3 frame impossible close absent; reconstruct via API
  d=get_stock_daily_data(s,4000)
  if d is None:d=get_index_daily_data(s,4000)
  d=d.sort_values('date'); ys.append(pd.DataFrame({'date':pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'),'symbol':s,'r3':d.close/d.close.shift(3)-1,'fr':d.close.shift(-h)/d.close-1}))
 q=pd.concat(ys); q['f']=-(q.r3-q.groupby('date').r3.transform('mean')); vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8 and g.f.nunique()>1: vals.append(g.f.corr(g.fr));ns.append(len(g))
 a=pd.Series(vals).dropna(); print('h',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),4),'hit',round((a>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4),'recent500',round(a.tail(500).mean(),6))
