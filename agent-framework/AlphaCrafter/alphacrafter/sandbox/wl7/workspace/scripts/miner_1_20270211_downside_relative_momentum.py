import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-10')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x['date']=pd.to_datetime(x['date']).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
# Explicit aligned matrices avoid nested concat/alignment artifacts.
P=pd.DataFrame({s:d['close'] for s,d in D.items()}).sort_index()
ret=P.pct_change(fill_method=None)
R=P.pct_change(10,fill_method=None)
down=ret.where(ret<0).rolling(30,min_periods=10).std()
# relative momentum score: positive means above peer median, risk-normalized; lag one bar.
F=R.sub(R.median(axis=1),axis=0).div(down.replace(0,np.nan)).shift(1)
FR=P.shift(-1).div(P)-1

def run(lo=None,hi=None,h=1):
 a=[]; ns=[]
 for dt in F.index:
  if lo and dt<pd.Timestamp(lo): continue
  if hi and dt>pd.Timestamp(hi): continue
  q=pd.concat([F.loc[dt],FR.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
 a=pd.Series(a)
 return len(a),float(np.mean(ns)) if ns else np.nan,float(a.mean()) if len(a) else np.nan,float(a.mean()/a.std(ddof=1)*np.sqrt(252)) if len(a)>1 and a.std(ddof=1)>0 else np.nan,float((a>0).mean()) if len(a) else np.nan
print('assets',len(D),'dates',P.index.min().date(),P.index.max().date(),'overall',run(),'coverage',float(F.notna().mean().mean()))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-02-10'),('2026-07-16','2027-02-10')]: print(lo+' '+hi,run(lo,hi))
print('turnover',float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
F.rank(axis=1,pct=True).to_csv('scripts/miner_1_20270211_downside_relative_momentum_signal.csv')
