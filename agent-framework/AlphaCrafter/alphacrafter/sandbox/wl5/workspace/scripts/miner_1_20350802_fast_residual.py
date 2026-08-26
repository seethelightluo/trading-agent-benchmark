import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); M=R.mean(axis=1)
b=R.rolling(60,min_periods=40).cov(M).div(M.rolling(60,min_periods=40).var(),axis=0)
res5=P.pct_change(5)-b.mul(M.rolling(5,min_periods=4).sum(),axis=0)
idv=R.sub(b.mul(M,axis=0),axis=0).rolling(20,min_periods=15).std()*np.sqrt(5)
# Fast residual reversal, strengthened when the asset has positive medium-term residual trend.
trend=P.pct_change(60)-b.mul(M.rolling(60,min_periods=40).sum(),axis=0)
sig=(-res5/(idv+1e-12))*(1+0.55*np.tanh(trend/(idv+1e-12)))
print('assets',len(P.columns),'rows',len(P),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 q=P.shift(-h)/P-1; aa=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): aa.append(c);ds.append(dt);ns.append(len(z))
 aa=np.array(aa); ds=pd.DatetimeIndex(ds); ns=np.array(ns)
 print('horizon',h,'dates',len(aa),'mean_n',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6),'hit',round((aa>0).mean(),6))
 if h==10 and len(aa):
  for a,bnd in [('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-08-01')]:
   w=aa[(ds>=a)&(ds<=bnd)];print('regime',a,bnd,'dates',len(w),'IC',round(w.mean(),6) if len(w) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds);print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20350802_fast_residual_signal.csv',index=False)
