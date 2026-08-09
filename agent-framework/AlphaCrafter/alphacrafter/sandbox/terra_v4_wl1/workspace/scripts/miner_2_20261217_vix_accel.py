import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END].copy()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'acc':d.close.pct_change(5)-d.close.pct_change(20),'y10':d.close.shift(-10)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date');v=v[v.date<=END].set_index('date').close
# lagged VIX regime, known at signal date; rolling stats use through date
vz=(v-v.rolling(60,min_periods=30).mean())/v.rolling(60,min_periods=30).std()
reg=(vz>0.5).astype(float)
med=x.pivot(index='date',columns='symbol',values='acc').median(axis=1)
x['factor']=-(x.acc-x.date.map(med))*(1+0.75*x.date.map(reg))
def calc(z):
 a=[];ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['factor','y10'])
  if len(g)>=8:
   q=spearmanr(g.factor,g.y10).statistic
   if np.isfinite(q):a.append(q);ns.append(len(g))
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('full',calc(x))
for lo,hi,n in [('2020-01-01','2022-12-31','2020-22'),('2023-01-01','2024-12-31','2023-24'),('2025-01-01','2026-12-17','2025-26')]: print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
vv=x.dropna(subset=['factor']); r=vv.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(vv)/len(x),'turnover',r.diff().abs().mean(axis=1).mean(),'highvixdays',reg.mean(),'symbols',x.symbol.nunique())
vv[['date','symbol','factor']].to_csv('scripts/miner_2_20261217_vix_accel_signal.csv',index=False)
