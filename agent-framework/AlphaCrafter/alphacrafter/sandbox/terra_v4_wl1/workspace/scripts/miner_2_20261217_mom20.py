import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END].set_index('date')
 r=d.close.pct_change()
 # lagged 20-day return, demeaned cross-section on each date, risk adjusted by lagged 60d volatility
 mom=r.rolling(20,min_periods=15).sum().shift(1)
 vol=r.rolling(60,min_periods=30).std().shift(1)
 rows.append(pd.DataFrame({'date':d.index,'symbol':s,'raw':mom,'vol':vol,'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
x['csmed']=x.groupby('date')['raw'].transform('median')
x['factor']=(x.raw-x.csmed)/x.vol

def calc(z):
 vals=[]; ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8:
   q=spearmanr(g.factor,g.y).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(g))
 a=np.asarray(vals)
 return {'dates':len(a),'avg_n':float(np.mean(ns)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean())}
print('candidate: 20d demeaned momentum / 60d volatility; rows',len(x))
print('all',calc(x))
for lo,hi,n in [('2020-01-01','2022-12-31','2020-22'),('2023-01-01','2024-12-31','2023-24'),('2025-01-01','2026-12-17','2025-26')]: print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
for h in [5,10]:
 z=[]
 for s in syms:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END]
  z.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 print('horizon',h,calc(x[['date','symbol','factor']].merge(pd.concat(z),on=['date','symbol'])))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',float(ranks.diff().abs().mean(axis=1).mean()))
v[['date','symbol','factor']].to_csv('scripts/miner_2_20261217_mom20_signal.csv',index=False)
