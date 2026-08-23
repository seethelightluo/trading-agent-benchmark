import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-12')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:load(s) for s in U};D={s:x for s,x in D.items() if x is not None}
P=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index();r=P.pct_change(fill_method=None)
# Volatility-shock reversal: fade lagged 3-day move, with recent volatility expansion
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
F=(-r.rolling(3,min_periods=3).sum()/(v20*np.sqrt(3)+1e-12)*(v20/(v60+1e-12))).shift(1)
def ev(h,lo=None,hi=None):
 Y=P.shift(-h).div(P)-1;a=[];ns=[]
 for dt in F.index:
  if lo and dt<pd.Timestamp(lo):continue
  if hi and dt>pd.Timestamp(hi):continue
  q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 a=np.array(a);return len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1)*np.sqrt(252),4),round(np.mean(a>0),4)
print('assets',len(D),'dates',P.index.min().date(),P.index.max().date())
for h in [1,5,10,20]:
 print('horizon',h,'overall',ev(h))
 for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-02-12'),('2026-07-16','2027-02-12')]:print(lo[:4]+'-'+hi[:4],ev(h,lo,hi))
print('coverage',round(float(F.notna().mean().mean()),4),'turnover',round(float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),4))
F.to_csv('scripts/miner_1_20270212_volshock_reversal_signal.csv')
