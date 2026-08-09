import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); v=d.volume.replace(0,np.nan)
 shock=(v.shift(1)/(v.shift(2).rolling(20,min_periods=10).median()+1e-12)-1).clip(lower=0)
 d['factor']=d.close.shift(1).pct_change(20)*np.log1p(shock)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); print('universe',len(U),'period',x.date.min().date(),x.date.max().date())
for h in [1,5,10]:
 a=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1: a.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 q=pd.Series(a); print('H',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 if h==1:
  for yr,z in q.groupby(pd.Series([x for x in []])): pass
r=x.dropna(subset=['factor']); print('coverage',len(r)/len(x)); ranks=r.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean())
print('corr plain mom',r.factor.corr(r.groupby('symbol').factor.transform('mean')))
