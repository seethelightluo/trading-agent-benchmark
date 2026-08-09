import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 # two-day average signed intraday location, reversed; range normalized and volatility scaled
 rng=(d.high-d.low).replace(0,np.nan)
 clv=((d.close-d.open)/rng).clip(-1,1)
 f=-(clv.rolling(2,min_periods=2).mean())
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor':f,'y1':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1,'y10':d.close.shift(-10)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
def calc(df,col):
 a=[]; ns=[]
 for dt,g in df.groupby('date'):
  g=g.dropna(subset=['factor',col])
  if len(g)>=8:a.append(spearmanr(g.factor,g[col]).statistic);ns.append(len(g))
 a=np.asarray(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
for col in ['y1','y5','y10']:
 q=calc(x,col); print(col,'dates',q[0],'avgN',round(q[1],2),'IC',round(q[2],6),'ICIR',round(q[3],6),'hit',round(q[4],4))
for lo,hi,name in [('2020-01-01','2022-12-31','2020-22'),('2023-01-01','2024-12-31','2023-24'),('2025-01-01','2026-12-17','2025-26')]:
 q=calc(x[(x.date>=lo)&(x.date<=hi)],'y1');print(name,'dates',q[0],'IC',round(q[2],6),'ICIR',round(q[3],6))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('coverage',round(len(v)/len(x),4),'turnover',round(ranks.diff().abs().mean(axis=1).mean(),6),'artifact_rows',len(v)); v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_clv2_signal.csv',index=False)
