import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is not None and len(d)>=220: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); mkt=r.mean(axis=1)
res=r.sub(mkt,axis=0)
# medium-term residual trend, lagged one observation and volatility normalized
trend=res.rolling(120,min_periods=90).sum().shift(1)
vol=res.rolling(40,min_periods=30).std().shift(1)
sig=(trend/(vol*np.sqrt(120)+1e-12)).clip(-8,8)
Q=P.shift(-10)/P-1
ic=[]; dates=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],Q.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ic.append(c);dates.append(dt);ns.append(len(z))
a=np.asarray(ic); dates=pd.DatetimeIndex(dates)
print('dates',len(a),'start',dates[0].date(),'end',dates[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
for x,y in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-09-01')]:
 z=a[(dates>=pd.Timestamp(x))&(dates<=pd.Timestamp(y))]
 if len(z)>1: print('regime',x,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round(np.mean(z>0),4))
S=sig.rank(pct=True); print('turnover',round(S.diff().abs().mean().mean(),6))
for h in [5,10,20]:
 q=P.shift(-h)/P-1; aa=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): aa.append(c)
 print('decay',h,round(np.mean(aa),6),round(np.mean(aa)/np.std(aa,ddof=1),6))
pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20340928_residual_momentum120_signal.csv',index=False)
