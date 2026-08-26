import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}; absret={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=100:
  x=d.set_index('date'); cl[s]=x.close.astype(float); absret[s]=x.close.astype(float).pct_change().abs()
P=pd.DataFrame(cl).sort_index(); A=pd.DataFrame(absret).reindex(P.index); R=P.pct_change();
# Directional efficiency: signed displacement divided by total path length. Causal and interpretable.
sig=R.rolling(30,min_periods=24).sum()/(A.rolling(30,min_periods=24).sum()+1e-12)
print('assets',len(P.columns),'rows',len(P),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 q=P.shift(-h)/P-1; aa=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): aa.append(c);ds.append(dt);ns.append(len(z))
 aa=np.array(aa); ds=pd.DatetimeIndex(ds); ns=np.array(ns)
 if len(aa): print('horizon',h,'dates',len(aa),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6),'hit',round((aa>0).mean(),6))
 if h==10 and len(aa):
  for a,b in [('2023-11-13','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-08-15')]:
   w=aa[(ds>=a)&(ds<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.mean(),6) if len(w) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds); print('turnover',round(ranks.diff().abs().mean().mean(),6))
  out=pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value'])
  out.to_csv('scripts/miner_2_20350816_directional_efficiency_signal.csv',index=False)
