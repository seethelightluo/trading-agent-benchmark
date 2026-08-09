import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Three-day close-to-close reversal: negative recent return, tested against next-day return.
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')[['date','close']]
 d['f']=-(d.close/d.close.shift(3)-1); d['symbol']=s; rows.append(d)
x=pd.concat(rows,ignore_index=True)
for h in [1,5,10]:
 q=[]
 for s,g in x.groupby('symbol'):
  g=g.sort_values('date').copy();g['y']=g.close.shift(-h)/g.close-1;q.append(g)
 q=pd.concat(q); ic=[];ds=[];ns=[]
 for dt,g in q.groupby('date'):
  g=g.dropna(subset=['f','y'])
  if len(g)>=8: ic.append(spearmanr(g.f,g.y).statistic);ds.append(dt);ns.append(len(g))
 a=np.array(ic); print('h',h,'dates',len(a),'avg_names',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
 for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2027-12-31')]:
  z=a[(np.array(ds)>=pd.Timestamp(lo))&(np.array(ds)<=pd.Timestamp(hi))]
  if len(z)>1: print(' ',lo[:4]+'-'+hi[:4],len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1))
wide=x.pivot(index='date',columns='symbol',values='f');print('coverage',x.f.notna().mean(),'turnover',wide.rank(axis=1,pct=True).diff().abs().mean().mean())
x[['date','symbol','f']].dropna().to_csv('scripts/miner_1_20270114_reversal3_signal.csv',index=False)
