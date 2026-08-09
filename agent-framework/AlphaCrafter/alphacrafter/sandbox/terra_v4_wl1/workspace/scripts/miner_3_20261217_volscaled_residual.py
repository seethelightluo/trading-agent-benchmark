import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 r=d.close.pct_change(); rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r3':d.close.pct_change(3),'vol20':r.rolling(20,min_periods=10).std(),'y1':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True); w=x.pivot(index='date',columns='symbol',values='r3'); med=w.median(axis=1); med[w.count(axis=1)<8]=np.nan
x['factor']=-(x.r3-x.date.map(med))/x.vol20.replace(0,np.nan)
def calc(df):
 a=[]; ns=[]
 for dt,g in df.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8:a.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
 a=np.asarray(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
x['y']=x.y1; print('H1 dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.4f'%calc(x))
for lo,hi,name in [('2020-01-01','2022-12-31','2020-22'),('2023-01-01','2024-12-31','2023-24'),('2025-01-01','2026-12-17','2025-26')]:
 z=x[(x.date>=lo)&(x.date<=hi)]; q=calc(z); print(name,'dates',q[0],'IC',round(q[2],6),'ICIR',round(q[3],6))
for h in [5,10]:
 rows2=[]
 for s in syms:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];rows2.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 z=x[['date','symbol','factor']].merge(pd.concat(rows2),on=['date','symbol']);q=calc(z);print('H'+str(h),'dates',q[0],'IC',round(q[2],6),'ICIR',round(q[3],6))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('coverage',round(len(v)/len(x),4),'turnover',round(ranks.diff().abs().mean(axis=1).mean(),6),'artifact_rows',len(v));v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_volscaled_residual_signal.csv',index=False)
