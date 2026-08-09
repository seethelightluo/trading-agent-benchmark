import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
parts=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]
 parts.append(pd.DataFrame({'date':d.date,'symbol':s,'r5':d.close.pct_change(5),'y1':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1,'y10':d.close.shift(-10)/d.close-1}))
x=pd.concat(parts,ignore_index=True)
# lagged 20d cross-asset dispersion, activate above rolling 120d 75th percentile
rets=x.pivot(index='date',columns='symbol',values='r5'); disp=rets.std(axis=1); threshold=disp.rolling(120,min_periods=60).quantile(.75).shift(1)
active=(disp.shift(1)>threshold)
med=rets.median(axis=1).shift(1)
x['factor']=-(x.r5-x.date.map(med))*x.date.map(active).astype(float)
def calc(y):
 z=x.assign(target=y); a=[]; ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['factor','target'])
  if len(g)>=8:a.append(spearmanr(g.factor,g.target).statistic);ns.append(len(g))
 a=np.array(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('UNIVERSE',len(syms),'DATES',x.date.nunique(),'active_days',int(active.sum()))
for h in ['y1','y5','y10']: print(h,calc(h))
for lo,hi,n in [('2020','2022','2020-22'),('2023','2024','2023-24'),('2025','2026-12-17','2025-26')]:
 z=x[(x.date>=lo)&(x.date<=hi)].copy()
 a=[];ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['factor','y1'])
  if len(g)>=8:a.append(spearmanr(g.factor,g.y1).statistic);ns.append(len(g))
 a=np.array(a);print(n,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean(),'artifact_rows',len(v))
v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_dispersion_reversal5_signal.csv',index=False)
