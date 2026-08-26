import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000300.SH','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; U=list(dict.fromkeys(U))
D={s:get_stock_daily_data(s,3100).set_index('date') for s in U}; C=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); R=C.pct_change(); M=R.mean(axis=1)
cov=R.rolling(60,min_periods=40).cov(M); var=M.rolling(60,min_periods=40).var(); E=R-cov.div(var,axis=0).mul(M,axis=0)
vol=E.rolling(20,min_periods=15).std()*np.sqrt(20); trend=E.rolling(30,min_periods=24).sum()/(vol+1e-12); eff=E.rolling(30,min_periods=24).sum().abs()/(E.abs().rolling(30,min_periods=24).sum()+1e-12); sig=-trend*(0.5+eff.clip(0,1))
print('assets',len(C.columns),'rows',len(C),'start',C.index.min().date(),'end',C.index.max().date())
for h in [5,10,20]:
 q=C.shift(-h)/C-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); ds.append(dt); ns.append(len(z))
 a=np.array(vals); ds=pd.DatetimeIndex(ds); ns=np.array(ns); print('h',h,'dates',len(a),'meanN',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),6))
 if h==10:
  for aa,bb in [('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-09-12')]:
   w=a[(ds>=aa)&(ds<=bb)]; print('regime',aa,bb,len(w),round(w.mean(),6) if len(w) else None)
  print('turnover',round(pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds).diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20350913_residual_trend_efficiency_signal.csv',index=False)
