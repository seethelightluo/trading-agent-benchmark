import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=2600)
 if d is not None: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Medium-term trend acceleration: recent 20d return minus average daily return over prior 60d.
# Lag one completed bar; cross-sectional rank is not used in IC calculation.
f=(r.rolling(20,min_periods=20).sum()-r.rolling(60,min_periods=60).sum()/3).shift(1)
def ev(Y,sl=slice(None)):
 a=[]; ns=[]
 for dt in f.loc[sl].index:
  z=pd.concat([f.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q);ns.append(len(z))
 a=np.asarray(a); return len(a),round(float(np.mean(ns)),2),round(float(np.mean(a)),6),round(float(np.mean(a)/np.std(a,ddof=1)),6),round(float(np.mean(a>0)),4)
for h in [1,3,5,10]: print('h',h,ev(np.log(p).shift(-h)-np.log(p)))
rank=f.rank(pct=True,axis=1); print('coverage',round(float(f.notna().sum(axis=1).mean()/15),4),'turnover',round(float(rank.diff().abs().mean(axis=1).mean()),5),'dates',len(p),'instruments',len(D))
for n,s in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]: print(n,ev(np.log(p).shift(-1)-np.log(p),s))
