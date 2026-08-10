import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END]
 r=d.close.pct_change(); rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r3':d.close.pct_change(3),'v20':r.rolling(20,min_periods=15).std(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True); p=x.pivot(index='date',columns='symbol',values='r3'); med=p.median(axis=1)
ix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:END]
v=ix.close.reindex(p.index).ffill(); vr=(v/v.rolling(120,min_periods=60).median()).clip(.5,2.0)
x['factor']=-(x.r3-x.date.map(med))/x.v20*x.date.map(vr.shift(1))
def calc(z):
 a=[];ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8:
   a.append(g.factor.corr(g.y,method='spearman'));ns.append(len(g))
 a=pd.Series(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('universe',len(U),'rows',len(x));print('H1',calc(x))
for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026-12-17','25-26')]:print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
for h in [5,10]:
 z=[]
 for s in U:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];z.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 print('H',h,calc(x[['date','symbol','factor']].merge(pd.concat(z),on=['date','symbol'])))
vv=x.dropna(subset=['factor']); ranks=vv.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(vv)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean(),'artifact_rows',len(vv))
vv[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_vix_residual_signal.csv',index=False)
