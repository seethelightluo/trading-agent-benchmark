import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={s:get_stock_daily_data(s,3000).set_index('date').close.astype(float) for s in U}; P=pd.DataFrame(px).sort_index(); R=P.pct_change(); M=R.mean(axis=1)
beta=R.rolling(60,min_periods=40).cov(M).div(M.rolling(60,min_periods=40).var(),axis=0); res=P.pct_change(20)-beta.mul(M.rolling(20).sum(),axis=0); idvol=R.sub(beta.mul(M,axis=0),axis=0).rolling(20,min_periods=15).std()*np.sqrt(20); base=-res/(idvol+1e-12)
vd=get_index_daily_data('VIX',3000); v=vd.set_index('date').close.astype(float).reindex(P.index).ffill(); vm=v.rolling(252,min_periods=120).mean(); vs=v.rolling(252,min_periods=120).std(); stress=((v-vm)/(vs+1e-12)).clip(-2,2)/4; sig=base.mul((1+.55*stress).clip(.65,1.55),axis=0)
print('assets',len(P.columns),'rows',len(P),'dates',P.index.min().date(),P.index.max().date(),'signal_nonnull',sig.notna().sum().sum())
for h in [5,10,20]:
 q=P.shift(-h)/P-1;a=[];ds=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):a.append(c);ds.append(dt);ns.append(len(z))
 a=np.array(a);ds=pd.DatetimeIndex(ds);ns=np.array(ns); print('horizon',h,'dates',len(a),'mean_n',round(ns.mean(),3) if len(ns) else None,'coverage',round(ns.mean()/15,6) if len(ns) else None,'IC',round(a.mean(),6) if len(a) else None,'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None,'hit',round((a>0).mean(),6) if len(a) else None)
 if h==10 and len(a):
  for x,y in [('2023-11-13','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-08-01')]:
   w=a[(ds>=x)&(ds<=y)];print('regime',x,'dates',len(w),'IC',round(w.mean(),6) if len(w) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds);print('turnover',round(ranks.diff().abs().mean().mean(),6));pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20350802_vix_stress_residual_reversal_signal.csv',index=False)
