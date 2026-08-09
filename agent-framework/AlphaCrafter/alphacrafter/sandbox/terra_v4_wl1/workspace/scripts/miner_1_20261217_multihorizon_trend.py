import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 r=d.close.pct_change()
 # multi-horizon trend consistency, all inputs lagged one day at decision
 d['factor']=(d.close.shift(1).pct_change(5)+d.close.shift(1).pct_change(20)+d.close.shift(1).pct_change(60))/3
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True); out={}
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1 and g[f'y{h}'].nunique()>1:
   ic=spearmanr(g.factor,g[f'y{h}']).statistic
   if np.isfinite(ic): a.append((dt,ic,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']); q=z.ic
 print('H',h,'dates',len(q),'avgN',round(z.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr,g in z.groupby(z.date.dt.year): print('YR',yr,'dates',len(g),'IC',round(g.ic.mean(),5),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),4))
 out[h]=z
v=x.dropna(subset=['factor']); rank=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',round(len(v)/len(x),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
x.to_csv('scripts/miner_1_20261217_multihorizon_trend_signal.csv',index=False)
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
