import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}; P=pd.DataFrame({s:D[s].close for s in U}).sort_index(); V=pd.DataFrame({s:D[s].volume for s in U}).reindex(P.index)
# Lagged volume-confirmed 20d momentum, evaluated against next completed day.
F=P.pct_change(20)*np.log1p((V.shift(1)/(V.shift(2).rolling(20,min_periods=10).median()+1e-12)-1).clip(lower=0)); Y=P.shift(-1).div(P)-1
vals=[]; ns=[]; dates=[]
for d in P.index:
 q=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: vals.append(spearmanr(q.f,q.y).statistic);ns.append(len(q));dates.append(d)
a=np.asarray(vals); print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=pd.Series(a,index=pd.to_datetime(dates)).loc[lo:hi+'-12-31'];print('regime',lo,hi,len(z),z.mean(),z.mean()/z.std(ddof=1))
