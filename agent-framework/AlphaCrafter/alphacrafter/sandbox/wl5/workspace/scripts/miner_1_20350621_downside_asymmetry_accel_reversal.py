import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3200)
 if d is not None and len(d)>=220: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); cs=r.sub(r.mean(axis=1),axis=0)
rev=(-cs.rolling(10,min_periods=8).sum()/(cs.rolling(40,min_periods=25).std()*np.sqrt(10)+1e-12))
# robust downside/upside amplitude ratio, requiring only 5 observations in each tail
neg=r.where(r<0); pos=r.where(r>0)
down=neg.abs().rolling(20,min_periods=5).mean(); up=pos.rolling(20,min_periods=5).mean()
asym=(down/(up+1e-12)).replace([np.inf,-np.inf],np.nan).clip(.5,3.0)
acc=(asym/asym.shift(5)-1).clip(-1,1)
sig=(rev*(1+0.75*acc)).clip(-8,8)
print('assets',len(P.columns),'rows',len(P),'date_start',P.index.min().date(),'date_end',P.index.max().date())
for h in [5,10,20]:
 q=P.shift(-h)/P-1; a=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): a.append(c);ds.append(dt);ns.append(len(z))
 a=np.asarray(a); ds=pd.DatetimeIndex(ds); ns=np.asarray(ns)
 print('horizon',h,'dates',len(a),'start',ds[0].date() if len(ds) else None,'end',ds[-1].date() if len(ds) else None,'mean_n',round(ns.mean(),3) if len(ns) else None,'coverage',round(ns.mean()/len(U),6) if len(ns) else None,'IC',round(a.mean(),6) if len(a) else None,'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None,'hit',round((a>0).mean(),6) if len(a) else None)
 if h==10 and len(a):
  for x,y in [('2023-09-19','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-06-20')]:
   w=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]; print('regime',x,y,'dates',len(w),'IC',round(w.mean(),6) if len(w) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds); print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20350621_downside_asymmetry_accel_reversal_signal.csv',index=False)
