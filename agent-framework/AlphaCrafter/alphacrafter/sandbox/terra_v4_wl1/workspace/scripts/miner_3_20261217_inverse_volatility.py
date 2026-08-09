import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); r=d.close.pct_change()
 # lagged inverse 20d realized volatility: favor quieter assets cross-sectionally
 d['factor']=-(r.rolling(20,min_periods=15).std().shift(1))
 d['fwd']=d.close.shift(-1)/d.close-1
 rows.append(d[['date','factor','fwd']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True); a=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.factor.nunique()>1: a.append(spearmanr(g.factor,g.fwd).statistic); ns.append(len(g))
a=pd.Series(a); print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('coverage',x.factor.notna().mean()); q=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',q.diff().abs().mean(axis=1).mean()); print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
print(x.assign(year=x.date.dt.year).groupby('year').apply(lambda g: spearmanr(g.dropna().factor,g.dropna().fwd).statistic if len(g.dropna())>10 else np.nan))
