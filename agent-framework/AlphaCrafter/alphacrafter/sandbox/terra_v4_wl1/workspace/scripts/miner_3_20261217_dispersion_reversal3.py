import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
parts=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]
 parts.append(pd.DataFrame({'date':d.date,'symbol':s,'r3':d.close.pct_change(3),'y1':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1,'y10':d.close.shift(-10)/d.close-1}))
x=pd.concat(parts,ignore_index=True); rets=x.pivot(index='date',columns='symbol',values='r3'); disp=rets.std(axis=1); threshold=disp.rolling(120,min_periods=60).median().shift(1); active=(disp.shift(1)>threshold); med=rets.median(axis=1).shift(1); x['factor']=-(x.r3-x.date.map(med)); x['active']=x.date.map(active).fillna(False)
def calc(y,df):
 a=[];ns=[]
 for dt,g in df.groupby('date'):
  g=g.dropna(subset=['factor',y])
  if len(g)>=8 and g.factor.nunique()>1 and g[y].nunique()>1:a.append(spearmanr(g.factor,g[y]).statistic);ns.append(len(g))
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('UNIVERSE',len(syms),'DATES',x.date.nunique(),'active_days',int(active.sum()))
for h in ['y1','y5','y10']: print(h,calc(h,x[x.active]))
for lo,hi,n in [('2020','2022','2020-22'),('2023','2024','2023-24'),('2025','2026-12-17','2025-26')]: print(n,calc('y1',x[x.active&(x.date>=lo)&(x.date<=hi)]))
v=x[x.active].dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('coverage_active',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean(),'artifact_rows',len(v));v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_dispersion_reversal3_signal.csv',index=False)
