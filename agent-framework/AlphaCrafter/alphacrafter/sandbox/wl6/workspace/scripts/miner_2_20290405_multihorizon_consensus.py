import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=3000)
   if x is not None and len(x): return x
  except Exception: pass
raw={s:fetch(s) for s in U}; p=pd.DataFrame({s:x.set_index('date')['close'] for s,x in raw.items() if x is not None}).sort_index(); r=p.pct_change()
# Consensus trend: rank-average of 10/30/60d returns, damped when horizons disagree.
rets=[p.pct_change(n) for n in (10,30,60)]
ranked=[x.rank(axis=1,pct=True) for x in rets]
base=sum(ranked)/3
sign=(sum((x>0).astype(float) for x in rets)/3)
agreement=1-abs(sign-0.5)*1.0 # keeps moderate disagreements from explosive effects
f=base*agreement
print('universe=%d dates=%d'%(len(p.columns),len(p)))
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: qs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(d)
 q=pd.Series(qs,index=pd.to_datetime(ds)); valid=f.loc[ds].notna().sum().sum()/(len(ds)*len(p.columns))
 print('h=%d dates=%d avg_n=%.2f coverage=%.4f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%(h,len(q),np.mean(ns),valid,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028'),('2029','2030')]:
  z=q[(q.index>=lo)&(q.index<=hi)]
  if len(z):print(' ',lo+'-'+hi,'n=%d IC=%.6f ICIR=%.6f'%(len(z),z.mean(),z.mean()/z.std(ddof=1)))
