import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-10')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception:pass
D={s:get(s) for s in U};D={s:x for s,x in D.items() if x is not None};P=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); r=P.pct_change(fill_method=None)
peer=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
for s in P.columns: peer[s]=r.rolling(5,min_periods=5).sum().drop(columns=[],errors='ignore').sub(r.rolling(5,min_periods=5).sum().median(axis=1),axis=0)[s] if False else np.nan
# leave-one-out peer median 5d return, normalized by own 20d realized volatility
R=r.rolling(5,min_periods=5).sum(); vol=r.rolling(20,min_periods=10).std()
for s in P.columns: peer[s]=R.drop(columns=s).median(axis=1)
F=peer.div(vol.replace(0,np.nan)).shift(1); FR=P.shift(-1).div(P)-1
def run(lo=None,hi=None):
 a=[];ns=[]
 for dt in F.index:
  if lo and dt<pd.Timestamp(lo):continue
  if hi and dt>pd.Timestamp(hi):continue
  q=pd.concat([F.loc[dt],FR.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 a=pd.Series(a);return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(252)),float((a>0).mean())
print('assets',len(D),'dates',P.index.min().date(),P.index.max().date(),'overall',run(),'coverage',float(F.notna().mean().mean()))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-02-10'),('2026-07-16','2027-02-10')]:print(lo+' '+hi,run(lo,hi))
print('turnover',float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()));F.rank(axis=1,pct=True).to_csv('scripts/miner_1_20270211_peer_leadlag_vol_signal.csv')
