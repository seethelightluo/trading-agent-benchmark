import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); rows=[]
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].copy()
 # Fade prior completed session's intraday move, with range normalization.
 intr=d.close/d.open-1
 rng=(d.high-d.low)/d.open
 d['factor']=(-intr/(rng.rolling(20,min_periods=10).median()+1e-9)).shift(1)
 d['y1']=d.close.shift(-1)/d.close-1; d['y5']=d.close.shift(-5)/d.close-1; d['y10']=d.close.shift(-10)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows)
for h in [1,5,10]:
 out=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8: out.append((dt,g.factor.corr(g[f'y{h}'],method='spearman'),len(g)))
 a=pd.DataFrame(out,columns=['date','ic','n']); q=a.ic
 print('H',h,'dates',len(q),'avgN',round(a.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  print('regimes',a.assign(reg=np.select([a.date.dt.year<=2022,a.date.dt.year<=2024],['2020-22','2023-24'],default='2025-26')).groupby('reg').ic.agg(['mean','count']).round(5).to_dict('index'))
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/(len(x)),4),'turnover',round(v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
v[['date','symbol','factor']].to_csv('scripts/miner_2_20261217_intraday_reversal_signal.csv',index=False)
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
