import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=3000)
   if d is not None and len(d)>=100:
    d=d.copy();d.date=pd.to_datetime(d.date);return d.set_index('date')
  except Exception: pass
p={s:get(s) for s in U};p={s:x for s,x in p.items() if x is not None}
C=pd.concat({s:x.close.astype(float) for s,x in p.items()},axis=1).sort_index().ffill(); R=C.pct_change()
# Range-expansion reversal: fade recent return when today's completed-session range is unusually large.
# Range uses only completed OHLC; signal is lagged one session relative to forward return.
H=pd.concat({s:x.high.astype(float) for s,x in p.items()},axis=1).reindex(C.index).ffill();L=pd.concat({s:x.low.astype(float) for s,x in p.items()},axis=1).reindex(C.index).ffill()
range_pct=(H-L)/C
shock=range_pct/(range_pct.rolling(30,min_periods=20).median()+1e-8)
raw=-R.rolling(3,min_periods=3).sum()*shock
f=raw.sub(raw.median(axis=1),axis=0).clip(-6,6)
print('cutoff',C.index.max().date(),'dates',len(C),'instruments',len(C.columns))
for h in [1,3,5,10]:
 fut=C.shift(-h)/C-1; vals=[];ds=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(q):vals.append(q);ds.append(dt);ns.append(len(z))
 ic=pd.Series(vals,index=ds);print('H',h,'obs',len(ic),'avgN',np.mean(ns),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1)*np.sqrt(len(ic)),'hit',(ic>0).mean())
 if h==1:
  print('coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
  for lab,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01',str(C.index.max().date()))]:
   x=ic.loc[a:b];print('REG',lab,len(x),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(len(x)))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270826_range_shock_reversal_signal.csv',index=False);print('artifact scripts/miner_2_20270826_range_shock_reversal_signal.csv');print('max_abs_library_correlation',None)
