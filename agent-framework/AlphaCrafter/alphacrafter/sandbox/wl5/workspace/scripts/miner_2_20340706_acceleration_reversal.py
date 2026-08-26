import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=100: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); P.index=pd.to_datetime(P.index).normalize(); R=P.pct_change()
# Acceleration exhaustion: recent 20d return minus prior 40d return, scaled by trailing volatility.
r20=P/P.shift(20)-1
prior40=P.shift(20)/P.shift(60)-1
vol=R.rolling(40,min_periods=25).std()*np.sqrt(20)
f=(-(r20-prior40)/(vol+1e-8)).replace([np.inf,-np.inf],np.nan).clip(-10,10)
fw=P.shift(-10)/P-1
a=[]; ds=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): a.append(c);ds.append(dt);ns.append(len(z))
a=np.asarray(a); ds=pd.DatetimeIndex(ds)
print('dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'daily_ICIR',round(a.mean()/a.std(ddof=1),6),'annualized_ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(a>0),6))
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2030-12-31'),('2031-01-01','2034-07-06')]:
 z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
 if len(z)>1: print('regime',x,y,'dates',len(z),'IC',round(z.mean(),6),'hit',round(np.mean(z>0),4))
S=pd.DataFrame([f.loc[d].rank(pct=True) for d in ds],index=ds)
print('turnover',round(S.diff().abs().mean().mean(),6))
rows=[(dt,s,float(f.loc[dt,s])) for dt in f.index for s in f.columns if pd.notna(f.loc[dt,s])]
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20340706_acceleration_reversal_signal.csv',index=False)
