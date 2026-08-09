import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
allr=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 d['r']=d.close.pct_change(); d['r3']=d.close.pct_change(3); d['vol']=d.r.rolling(20,min_periods=15).std()
 d['factor']=-(d['r3']/(d['vol']*np.sqrt(3)+1e-12)).shift(1)
 d['y1']=d.close.shift(-1)/d.close-1; d['y5']=d.close.shift(-5)/d.close-1
 allr.append(d[['date','factor','y1','y5']].assign(symbol=s))
x=pd.concat(allr)
cs=x.pivot(index='date',columns='symbol',values='y1')
disp=cs.std(axis=1).where(cs.count(axis=1)>=8).rolling(20,min_periods=10).mean().shift(1)
w=disp.rank(pct=True).clip(.25,.75)
x['factor']=x['factor']*x.date.map(w)
for h in ['y1','y5']:
 z=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',h])
  if len(g)>=8:z.append(spearmanr(g.factor,g[h]).statistic); ns.append(len(g))
 a=np.array(z); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(x.factor.notna().sum()/len(x),4),'period',x.date.min().date(),x.date.max().date(),'symbols',x.symbol.nunique())
# regimes
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8:obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(reg=pd.cut(o.date.dt.year,[2019,2022,2024,2026,2027])).groupby('reg',observed=True).ic.agg(['mean','count']).round(5).to_string())
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('turnover',round(r.diff().abs().mean(axis=1).mean(),5))
