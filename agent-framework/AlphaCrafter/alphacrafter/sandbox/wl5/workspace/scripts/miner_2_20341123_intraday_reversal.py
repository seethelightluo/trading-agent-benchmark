import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
o={};c={}
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is not None and len(d)>=140:
  x=d.set_index('date'); o[s]=x.open.astype(float); c[s]=x.close.astype(float)
O=pd.DataFrame(o).sort_index(); P=pd.DataFrame(c).sort_index()
# Fade the most recent close/open shock, normalized by trailing daily volatility.
r=P.pct_change(); intraday=P/O-1; v20=r.rolling(20,min_periods=15).std()
sig=(-intraday/(v20+1e-12)).clip(-6,6)
q=P.shift(-10)/P-1
ics=[]; ds=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  x=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(x): ics.append(x);ds.append(dt);ns.append(len(z))
a=np.asarray(ics); ds=pd.DatetimeIndex(ds)
print('dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
for h in [5,20]:
 q=P.shift(-h)/P-1; aa=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(x): aa.append(x)
 print('decay',h,round(np.mean(aa),6),'n',len(aa))
for x,y in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-11-22')]:
 z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
 if len(z)>1: print('regime',x,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds)
print('turnover',round(ranks.diff().abs().mean().mean(),6))
pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20341123_intraday_reversal_signal.csv',index=False)
