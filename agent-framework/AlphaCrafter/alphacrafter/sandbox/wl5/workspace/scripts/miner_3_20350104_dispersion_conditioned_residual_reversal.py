import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); common=r.mean(axis=1); res=r.sub(common,axis=0)
shock=res.rolling(10,min_periods=8).sum(); vol=res.rolling(40,min_periods=25).std()
base=(-shock/(vol*np.sqrt(10)+1e-12)).clip(-8,8)
disp=res.std(axis=1).rolling(20,min_periods=12).mean()
med=disp.rolling(252,min_periods=80).median()
# Smooth relative-dispersion multiplier, bounded to avoid unstable tail amplification.
mult=(disp/(med+1e-12)).clip(0.5,1.5)
sig=base*mult
print('assets',len(P.columns),'rows',len(P),'signal_dates',sig.notna().any(axis=1).sum())
q=P.shift(-10)/P-1; a=[]; ds=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): a.append(c);ds.append(dt);ns.append(len(z))
a=np.asarray(a); ds=pd.DatetimeIndex(ds)
print('dates',len(a),'start',ds[0].date() if len(ds) else None,'end',ds[-1].date() if len(ds) else None,'mean_n',round(np.mean(ns),3) if ns else None,'coverage',round(np.mean(ns)/15,6) if ns else None)
if len(a):
 print('IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
 for x,y in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2035-01-03')]:
  z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
  if len(z)>1: print('regime',x,len(z),round(z.mean(),6))
 ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds)
 print('turnover',round(ranks.diff().abs().mean().mean(),6))
 for h in [5,20]:
  f=P.shift(-h)/P-1; aa=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if np.isfinite(c): aa.append(c)
  print('decay',h,round(np.mean(aa),6),'dates',len(aa))
pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20350104_dispersion_conditioned_residual_reversal_signal.csv',index=False)
