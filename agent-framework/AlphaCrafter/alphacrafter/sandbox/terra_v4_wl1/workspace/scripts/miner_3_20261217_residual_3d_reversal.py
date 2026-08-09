import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]; prices=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); d['r3']=d.close.pct_change(3); rows.append(d[['date','r3']].assign(symbol=s))
 q=d[['date','close']].copy()
 for h in [1,5,10]: q[f'y{h}']=q.close.shift(-h)/q.close-1
 prices.append(q.drop(columns='close').assign(symbol=s))
x=pd.concat(rows); wide=x.pivot(index='date',columns='symbol',values='r3'); med=wide.median(axis=1); med[wide.count(axis=1)<8]=np.nan
x['factor']=-(x.r3-x.date.map(med)); y=pd.concat(prices); x=x.merge(y,on=['date','symbol'],how='left')
obs=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8: obs.append(spearmanr(g.factor,g.y1).statistic); ns.append(len(g))
a=np.array(obs); print('H1 dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for h in [5,10]:
 z=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8:z.append(spearmanr(g.factor,g[f'y{h}']).statistic)
 z=np.array(z); print('H'+str(h),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'dates',len(z))
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/len(x),4),'turnover',round(v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_residual_3d_signal.csv',index=False)
print('artifact_rows',len(v))
