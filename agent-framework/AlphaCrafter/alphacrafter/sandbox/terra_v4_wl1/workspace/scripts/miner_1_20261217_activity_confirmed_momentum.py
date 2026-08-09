import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END].copy(); r=d.close.pct_change()
 # Lagged volume shock times lagged medium-term momentum; interpretable activity-confirmed continuation.
 med=d.volume.shift(1).rolling(20,min_periods=10).median()
 shock=(d.volume.shift(1)/(med+1e-12)-1).clip(lower=0)
 d['factor']=d.close.pct_change(20).shift(1)*np.log1p(shock)
 for h in (1,5,10): d['y'+str(h)]=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in (1,5,10):
 out=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor','y'+str(h)])
  if len(g)>=8 and g.factor.nunique()>1 and g['y'+str(h)].nunique()>1:
   q=spearmanr(g.factor,g['y'+str(h)]).statistic
   if np.isfinite(q): out.append((dt,q,len(g)))
 z=pd.DataFrame(out,columns=['date','ic','n']); a=z.ic
 print('H',h,'dates',len(a),'avgN',round(z.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==1:
  print(z.assign(regime=pd.cut(z.date,[pd.Timestamp('2019-12-31'),pd.Timestamp('2022-12-31'),pd.Timestamp('2024-12-31'),END],labels=['2020-22','2023-24','2025-26'])).groupby('regime',observed=True).ic.agg(['mean','count']).round(6).to_string())
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/len(x),4))
r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('turnover',round(r.diff().abs().mean(axis=1).mean(),6))
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
x.to_csv('scripts/miner_1_20261217_activity_confirmed_momentum_signal.csv',index=False)
