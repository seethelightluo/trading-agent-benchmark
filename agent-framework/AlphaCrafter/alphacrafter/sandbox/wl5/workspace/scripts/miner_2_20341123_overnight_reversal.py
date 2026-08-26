import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; O={};P={}
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is not None and len(d)>=140:
  x=d.set_index('date'); O[s]=x.open.astype(float);P[s]=x.close.astype(float)
O=pd.DataFrame(O).sort_index();P=pd.DataFrame(P).sort_index(); r=P.pct_change(); gap=O/P.shift(1)-1; v20=r.rolling(20,min_periods=15).std(); sig=(-gap/(v20+1e-12)).clip(-6,6); Q=P.shift(-10)/P-1
A=[];D=[];N=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],Q.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  a=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(a):A.append(a);D.append(dt);N.append(len(z))
a=np.array(A);D=pd.DatetimeIndex(D);print('dates',len(a),'start',D[0].date(),'end',D[-1].date(),'mean_n',round(np.mean(N),3),'coverage',round(np.mean(N)/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
for h in [5,20]:
 q=P.shift(-h)/P-1;b=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:b.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 b=np.array([x for x in b if np.isfinite(x)]);print('decay',h,round(b.mean(),6))
for x,y in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-11-22')]:
 z=a[(D>=x)&(D<=y)]
 if len(z)>1:print('regime',x,len(z),round(z.mean(),6))
S=pd.DataFrame([sig.loc[d].rank(pct=True) for d in D]);print('turnover',round(S.diff().abs().mean().mean(),6))
pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20341123_overnight_reversal_signal.csv',index=False)
