import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,3100)
 if d is not None and len(d)>=140: D[s]=d.set_index('date')
C=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index()
H=pd.DataFrame({s:x.high.astype(float) for s,x in D.items()}).reindex(C.index); L=pd.DataFrame({s:x.low.astype(float) for s,x in D.items()}).reindex(C.index)
R=C.pct_change(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
# Range-expansion exhaustion reversal: reverse the causal 10d move, with
# stronger reversal for assets whose recent true range has abruptly expanded.
tr=(H-L)/C.replace(0,np.nan)
shock=(tr.rolling(5,min_periods=4).mean()/(tr.rolling(40,min_periods=25).mean()+1e-12)-1).clip(-0.6,1.5)
raw=-(C/C.shift(10)-1)/(vol+1e-12)
sig=raw*(1+0.8*shock)
print('assets',len(C.columns),'rows',len(C),'start',C.index.min().date(),'end',C.index.max().date())
for h in [5,10,20]:
 q=C.shift(-h)/C-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); ds.append(dt); ns.append(len(z))
 a=np.array(vals); ds=pd.DatetimeIndex(ds); ns=np.array(ns)
 if len(a): print('h',h,'dates',len(a),'meanN',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),6))
 if h==10 and len(a):
  for aa,bb in [('2023-11-13','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-08-29')]:
   w=a[(ds>=aa)&(ds<=bb)]; print('regime',aa,bb,len(w),round(w.mean(),6) if len(w) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds)
  print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20350830_range_expansion_signal.csv',index=False)
