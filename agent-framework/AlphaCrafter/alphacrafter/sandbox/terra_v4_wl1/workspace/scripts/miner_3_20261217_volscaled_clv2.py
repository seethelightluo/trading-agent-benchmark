import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 rng=(d.high-d.low).replace(0,np.nan)
 clv=((d.close-d.open)/rng).clip(-1,1)
 ret=d.close.pct_change()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor':-clv.rolling(2,min_periods=2).mean()/ret.rolling(20,min_periods=10).std(),'r1':d.close.shift(-1)/d.close-1,'r5':d.close.shift(-5)/d.close-1,'r10':d.close.shift(-10)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
def calc(z,y):
 a=[]; ns=[]
 z=z.assign(y=y)
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8:a.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
 a=np.asarray(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
for col in ['r1','r5','r10']:
 q=calc(x,x[col]);print(col,'dates',q[0],'avgN',round(q[1],2),'IC',round(q[2],6),'ICIR',round(q[3],6),'hit',round(q[4],4))
for lo,hi,n in [('2020-01-01','2022-12-31','2020-22'),('2023-01-01','2024-12-31','2023-24'),('2025-01-01','2026-12-17','2025-26')]:
 q=calc(x[(x.date>=lo)&(x.date<=hi)],x.loc[(x.date>=lo)&(x.date<=hi),'r1']);print(n,'dates',q[0],'IC',round(q[2],6),'ICIR',round(q[3],6))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',round(len(v)/len(x),4),'turnover',round(ranks.diff().abs().mean(axis=1).mean(),6),'artifact_rows',len(v))
v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_volscaled_clv2_signal.csv',index=False)
