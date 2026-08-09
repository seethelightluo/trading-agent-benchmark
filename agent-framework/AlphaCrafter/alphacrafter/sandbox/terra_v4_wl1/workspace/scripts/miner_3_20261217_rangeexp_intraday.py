import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
rows=[]
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut]
 prev=d.close.shift(1); intra=d.close/d.open-1; rng=(d.high-d.low)/prev
 med=rng.shift(1).rolling(20,min_periods=10).median(); expansion=(rng/med).clip(0.5,3.0)
 f=-(intra/med)*np.sqrt(expansion)
 for h in [1,5,10]: rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor':f,'h':h,'y':d.close.shift(-h)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
for h,g0 in x.groupby('h'):
 vals=[]; ns=[]
 for dt,g in g0.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8: vals.append(spearmanr(g.factor,g.y).statistic); ns.append(len(g))
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 if h==1:
  for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026','25-26')]:
   z=g0[(g0.date.dt.year>=int(lo))&(g0.date.dt.year<=int(hi))]; aa=[]
   for _,q in z.groupby('date'):
    q=q.dropna(subset=['factor','y'])
    if len(q)>=8: aa.append(spearmanr(q.factor,q.y).statistic)
   aa=np.array(aa); print(n,len(aa),aa.mean(),aa.mean()/aa.std(ddof=1))
v=x[x.h==1].dropna(subset=['factor']); r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('coverage',len(v)/len(x[x.h==1]),'turnover',r.diff().abs().mean(axis=1).mean(),'rows',len(v)); v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_rangeexp_intraday_signal.csv',index=False)
