import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=3000)
   if x is not None and len(x): return x
  except Exception: pass
raw={s:fetch(s) for s in U}
p=pd.DataFrame({s:x.set_index('date')['close'] for s,x in raw.items() if x is not None}).sort_index()
r=p.pct_change(); market=r.mean(axis=1)
# Residual strength: asset 10d return relative to contemporaneous cross-asset mean,
# scaled by trailing 20d idiosyncratic volatility; all inputs trailing through signal date.
res=r.sub(market,axis=0)
vol=res.rolling(20,min_periods=15).std()
f=res.rolling(10,min_periods=8).sum()/(vol*np.sqrt(10)+1e-12)
# stability overlay rewards consistent sign over trailing 20 observations
cons=np.sign(res).rolling(20,min_periods=15).mean()
f=f*(1+0.5*cons)
fr=p.shift(-10)/p-1
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: qs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(d)
 q=pd.Series(qs,index=pd.to_datetime(ds)); print('h=%d dates=%d avg_n=%.2f coverage=%.4f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(q),np.mean(ns),np.isfinite(f.loc[ds]).sum().sum()/(len(ds)*len(U)),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028'),('2029','2030')]:
  z=q[(q.index>=lo)&(q.index<=hi)]
  if len(z): print(' ',lo+'-'+hi,'n=%d IC=%.6f ICIR=%.6f'%(len(z),z.mean(),z.mean()/z.std(ddof=1)))
rank=f.rank(axis=1,pct=True); print('turnover=%.6f instruments=%d dates=%d'%(rank.diff().abs().mean(axis=1).mean(),len(p.columns),len(p)))
