import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); cs=r.sub(r.mean(axis=1),axis=0)
# residual trend: medium horizon relative return, risk scaled, with a causal breadth regime tilt
mom=cs.rolling(60,min_periods=40).sum()
vol=cs.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-12
base=(mom/vol).clip(-8,8)
breadth=(cs>0).mean(axis=1).rolling(10,min_periods=7).mean()
# damp trend in narrow breadth and modestly emphasize broad participation; all lagged
bstate=((breadth-breadth.rolling(60,min_periods=30).mean())/(breadth.rolling(60,min_periods=30).std()+1e-12)).clip(-2,2)
mult=(1+0.18*bstate).clip(0.65,1.35)
sig=base.mul(mult,axis=0).clip(-8,8)
print('assets',len(P.columns),'rows',len(P),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 q=P.shift(-h)/P-1; aa=[];ds=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): aa.append(c);ds.append(dt);ns.append(len(z))
 aa=np.asarray(aa); ds=pd.DatetimeIndex(ds); ns=np.asarray(ns)
 if len(aa): print('horizon',h,'dates',len(aa),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6),'hit',round((aa>0).mean(),6))
 if h==10 and len(aa):
  for x,y in [('2023-11-13','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-05-24')]:
   w=aa[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]; print('regime',x,y,'dates',len(w),'IC',round(w.mean(),6) if len(w) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds); print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20350524_residual_trend_breadth_signal.csv',index=False)
