import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; r=d.close.pct_change();
 # blend lagged 1d and 3d reversal, volatility normalized
 vol=r.rolling(20,min_periods=15).std().shift(1)
 f=(-(r.shift(1))*0.5-(d.close.pct_change(3).shift(1))*0.5)/(vol+1e-12)
 rows.append(pd.DataFrame({'date':d.date,'f':f,'y1':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1,'s':s}))
x=pd.concat(rows)
for h in ['y1','y5']:
 a=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['f',h])
  if len(g)>=8:a.append(spearmanr(g.f,g[h]).statistic);ns.append(len(g))
 a=np.array(a);print(h,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
print('coverage',x.f.notna().mean())
