import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=120: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); P.index=pd.to_datetime(P.index).normalize(); R=P.pct_change()
V=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float); V.index=pd.to_datetime(V.index).normalize(); V=V.reindex(P.index).ffill()
vchg=V.pct_change(10).clip(-2,2)
stress=((vchg-vchg.rolling(252,min_periods=60).mean())/(vchg.rolling(252,min_periods=60).std()+1e-8)).clip(-3,3)
base=-(P/P.shift(10)-1)/(R.rolling(30,min_periods=20).std()*np.sqrt(10)+1e-8)
f=base.mul((1+0.35*stress.clip(lower=0)),axis=0).replace([np.inf,-np.inf],np.nan).clip(-10,10)
fw=P.shift(-10)/P-1
a=[];ds=[];ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): a.append(c);ds.append(dt);ns.append(len(z))
a=np.array(a);ds=pd.DatetimeIndex(ds)
print('dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'daily_ICIR',round(a.mean()/a.std(ddof=1),6),'annualized_ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(a>0),6))
for x,y in [('2026-07-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-04-27')]:
 z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
 if len(z)>1: print('regime',x,y,'dates',len(z),'IC',round(z.mean(),6))
S=pd.DataFrame([f.loc[d].rank(pct=True) for d in ds],index=ds)
print('turnover',round(S.diff().abs().mean().mean(),6))
rows=[(dt,s,float(f.loc[dt,s])) for dt in f.index for s in f.columns if pd.notna(f.loc[dt,s])]
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20340427_vix_stress_reversal_signal.csv',index=False)
