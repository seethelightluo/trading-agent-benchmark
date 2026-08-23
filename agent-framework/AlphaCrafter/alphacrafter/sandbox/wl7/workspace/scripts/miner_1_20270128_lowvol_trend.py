import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-01-27')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
R=pd.concat({s:d.close.pct_change() for s,d in D.items()},axis=1); V=R.rolling(20).std(); M=pd.concat({s:d.close.pct_change(15) for s,d in D.items()},axis=1)
# Low-volatility quality with a mild trend confirmation, lagged one session.
F=(M/(V+1e-9)*0.4 + 1/(V+1e-9)*0.6).shift(1); FR=R.shift(-1)
vals=[]; ns=[]
for dt in F.index:
 q=pd.concat([F.loc[dt],FR.loc[dt]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
a=pd.Series(vals); print('assets',len(D),'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'hit',(a>0).mean(),'coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-27')]:
 z=[q for dt in F.loc[lo:hi].index if len(q:=pd.concat([F.loc[dt],FR.loc[dt]],axis=1).dropna())>=8 and q.iloc[:,0].nunique()>1]; b=pd.Series([q.iloc[:,0].corr(q.iloc[:,1],method='spearman') for q in z]); print(lo,b.mean(),b.mean()/b.std(ddof=1)*np.sqrt(252),len(b))
F.rank(axis=1,pct=True).to_csv('scripts/miner_1_20270128_lowvol_trend_signal.csv')
