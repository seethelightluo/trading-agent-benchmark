import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END].copy(); c=d.close
 # recovery signal: current drawdown from 60d high, scaled by 20d vol; more negative = deeper drawdown, reversal expects positive factor
 dd=c/c.rolling(60,min_periods=40).max()-1; vol=c.pct_change().rolling(20,min_periods=15).std()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor':-dd/(vol*np.sqrt(20)),'y10':c.shift(-10)/c-1}))
x=pd.concat(rows,ignore_index=True)
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
v=x.dropna(subset=['factor']);r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('coverage',len(v)/len(x),'turnover',r.diff().abs().mean(axis=1).mean(),'symbols',x.symbol.nunique())
v[['date','symbol','factor']].to_csv('scripts/miner_2_20261217_drawdown_recovery_signal.csv',index=False)
