import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]
 r=d.close.pct_change(); mom=d.close.pct_change(20); vol=r.rolling(20,min_periods=15).std()
 # medium-term momentum, risk adjusted and cross-sectional centered; all lagged at date t
 f=mom/vol.replace(0,np.nan)
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'f':f,'y':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1,'y10':d.close.shift(-10)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
def calc(z,ycol='y'):
 vals=[];ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['f',ycol])
  if len(g)>=8: vals.append(spearmanr(g.f,g[ycol]).statistic);ns.append(len(g))
 a=np.asarray(vals); return len(a),float(np.mean(ns)),float(np.mean(a)),float(np.mean(a)/np.std(a,ddof=1)),float((a>0).mean())
print('universe',len(syms),'rows',len(x))
for h in ['y','y5','y10']: print(h,calc(x,h))
for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026','25-26')]: print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
v=x.dropna(subset=['f']); ranks=v.pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean())
v[['date','symbol','f']].to_csv('scripts/miner_1_20261217_riskadj_momentum_signal.csv',index=False)
