import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is not None and len(d)>=140: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); r10=P/P.shift(10)-1
# cross-asset dispersion of trailing daily returns; normalize by its 120d median
D=r.rolling(20,min_periods=15).std().mean(axis=1)
Dn=D/(D.rolling(120,min_periods=80).median()+1e-12)
# elevated dispersion selectively strengthens short-horizon mean reversion
amp=Dn.clip(0.5,2.5).pow(.5)
sig=(-r10.mul(amp,axis=0)).clip(-6,6)
rows=[]; fwd={h:P.shift(-h)/P-1 for h in [5,10,20]}
for h,Q in fwd.items():
 ic=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],Q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): ic.append(c); dates.append(dt); ns.append(len(z))
 a=np.asarray(ic); dates=pd.DatetimeIndex(dates)
 if h==10:
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20340914_dispersion_conditioned_reversal_signal.csv',index=False)
  print('dates',len(a),'start',dates[0].date(),'end',dates[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
  for x,y in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-09-01')]:
   z=a[(dates>=pd.Timestamp(x))&(dates<=pd.Timestamp(y))]
   if len(z)>1: print('regime',x,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
  S=pd.DataFrame([sig.loc[d].rank(pct=True) for d in dates],index=dates); print('turnover',round(S.diff().abs().mean().mean(),6))
 print('decay',h,round(a.mean(),6))
