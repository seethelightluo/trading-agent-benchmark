import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); r=d.close.pct_change()
 d['factor']=d.close.pct_change(20)/(r.abs().rolling(20,min_periods=15).sum()+1e-12).shift(1)
 for h in [1,5,10]: d[f'fwd{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','fwd1','fwd5','fwd10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in [1,5,10]:
 a=[]; ns=[]
 for _,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'fwd{h}'])
  if len(g)>=8:
   z=spearmanr(g.factor,g[f'fwd{h}']).statistic
   if np.isfinite(z): a.append(z); ns.append(len(g))
 a=np.asarray(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
v=x.dropna(subset=['factor']); r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',round(len(v)/len(x),4),'turnover',round(r.diff().abs().mean(axis=1).mean(),6),'period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
