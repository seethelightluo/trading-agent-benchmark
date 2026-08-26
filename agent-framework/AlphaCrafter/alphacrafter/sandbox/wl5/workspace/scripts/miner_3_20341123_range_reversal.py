import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); ret10=P/P.shift(10)-1
v20=r.rolling(20,min_periods=15).std()
lo=P.rolling(120,min_periods=80).min(); hi=P.rolling(120,min_periods=80).max()
pos=((P-lo)/(hi-lo+1e-12)).clip(0,1)
# Fade recent moves, with a smooth boost for assets near the lower end of their 120d range.
# This remains cross-sectional and uses only information available at signal date.
gate=(1.5- pos).clip(.5,1.5)
sig=(-ret10/(v20*np.sqrt(10)+1e-12)*gate).clip(-8,8)
for h in [5,10,20]:
 Q=P.shift(-h)/P-1; a=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],Q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): a.append(c);ds.append(dt);ns.append(len(z))
 a=np.asarray(a); ds=pd.DatetimeIndex(ds)
 print('horizon',h,'dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
 if h==10:
  for x,y in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-11-22')]:
   z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
   if len(z)>1: print('regime',x,len(z),round(z.mean(),6))
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds)
  print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20341123_range_reversal_signal.csv',index=False)
