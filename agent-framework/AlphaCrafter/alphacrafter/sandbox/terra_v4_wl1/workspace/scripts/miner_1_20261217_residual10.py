import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; r=d.close.pct_change()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'raw':d.close.pct_change(10).shift(1),'vol':r.rolling(30,min_periods=20).std().shift(1),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows); p=x.pivot(index='date',columns='symbol',values='raw'); med=p.median(axis=1); x['factor']=-(x.raw-x.date.map(med))/x.vol
obs=[];ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y'])
 if len(g)>=8:obs.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
a=np.array(obs);print('universe',x.symbol.nunique(),'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026-12-17')]:
 q=x[(x.date>=lo)&(x.date<=hi)]; b=[]
 for dt,g in q.groupby('date'):
  g=g.dropna(subset=['factor','y']);
  if len(g)>=8:b.append(spearmanr(g.factor,g.y).statistic)
 b=np.array(b);print(lo,hi,'n',len(b),'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1))
v=x.dropna(subset=['factor']); print('coverage',len(v)/len(x)); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',ranks.diff().abs().mean(axis=1).mean());v[['date','symbol','factor']].to_csv('scripts/miner_1_20261217_residual10_signal.csv',index=False)
