import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].set_index('date'); r=d.close.pct_change();
 # lagged medium-term momentum, risk normalized, information only through t-1
 f=(r.rolling(60,min_periods=40).sum()/r.rolling(20,min_periods=15).std()).shift(1)
 rows.append(pd.DataFrame({'date':d.index,'symbol':s,'factor':f,'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
def calc(z):
 a=[]; ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna()
  if len(g)>=8:
   q=spearmanr(g.factor,g.y).statistic
   if np.isfinite(q):a.append(q);ns.append(len(g))
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('universe=15; rows',len(x)); print('H1',calc(x))
for lo,hi,n in [('2020','2022','2020-22'),('2023','2024','2023-24'),('2025','2026-12-17','2025-26')]: print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
for h in [5,10]:
 z=[]
 for s in syms:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];z.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 print('H'+str(h),calc(x[['date','symbol','factor']].merge(pd.concat(z),on=['date','symbol'])))
v=x.dropna(subset=['factor']); p=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',p.diff().abs().mean(axis=1).mean())
v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_medium_momentum_signal.csv',index=False)
