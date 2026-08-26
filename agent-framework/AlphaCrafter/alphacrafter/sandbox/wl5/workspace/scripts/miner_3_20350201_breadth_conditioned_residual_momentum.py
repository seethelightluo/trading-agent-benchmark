import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); res=r.sub(r.mean(axis=1),axis=0)
# Contrarian residual 20-session momentum, strengthened when breadth is broad.
base=(res.rolling(20,min_periods=15).sum()/(res.rolling(60,min_periods=35).std()*np.sqrt(20)+1e-12)).clip(-8,8)
ma=P.rolling(60,min_periods=35).mean(); breadth=(P>ma).mean(axis=1)
mult=(0.65+0.70*breadth).clip(.65,1.35)
sig=-base.mul(mult,axis=0)
def ev(h):
 q=P.shift(-h)/P-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); ds.append(dt); ns.append(len(z))
 return pd.Series(vals,index=pd.DatetimeIndex(ds)),pd.Series(ns,index=pd.DatetimeIndex(ds))
for h in [5,10,20]:
 ic,n=ev(h); print('horizon',h,'dates',len(ic),'start',ic.index[0].date(),'end',ic.index[-1].date(),'mean_n',round(n.mean(),3),'coverage',round(n.mean()/15,6),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),6))
 if h==10:
  for x,y in [('2023-09-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2030-12-31'),('2031-01-01','2035-01-31')]:
   z=ic.loc[x:y]; print('regime',x,y,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None)
  ranks=sig.loc[ic.index].rank(pct=True); print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20350201_breadth_conditioned_residual_momentum_signal.csv',index=False)
