import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=150: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); mr=R['SPX']
beta=R.rolling(60,min_periods=40).cov(mr).div(mr.rolling(60,min_periods=40).var(),axis=0)
resid=R.sub(beta.mul(mr,axis=0),axis=0); f=resid.rolling(20,min_periods=15).sum()
cs20=R.rolling(20,min_periods=15).sum(); disp=cs20.std(axis=1); q=disp.rolling(252,min_periods=100).quantile(.55)
f=f.where(disp.ge(q), -f*0.25)
fr=R.shift(-10).rolling(10,min_periods=10).sum(); ics=[]; dates=[]; nobs=[]; sig=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c); dates.append(dt); nobs.append(len(z)); sig.append(f.loc[dt].rank(pct=True))
ics=np.array(ics); mean=ics.mean(); sd=ics.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd>0 else np.nan
S=pd.DataFrame(sig,index=dates); turnover=S.diff().abs().mean().mean()
print({'dates':len(ics),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':float(np.mean(nobs)),'coverage':float(np.mean(nobs)/len(U)),'IC':float(mean),'ICIR':float(icir),'hit':float(np.mean(ics>0)),'turnover':float(turnover)})
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-05-11')]:
 z=ics[(np.array(dates)>=pd.Timestamp(a))&(np.array(dates)<=pd.Timestamp(b))]
 print(a,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)*np.sqrt(252)) if len(z)>1 and z.std(ddof=1)>0 else None)
