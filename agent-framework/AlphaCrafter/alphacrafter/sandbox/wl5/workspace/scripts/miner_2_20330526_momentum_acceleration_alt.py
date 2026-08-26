import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=150: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=np.log(P).diff(); f=pd.DataFrame(index=P.index,columns=P.columns)
for s in P:
 r=R[s]; f[s]=(r.rolling(20).sum()-r.shift(20).rolling(40).mean()*20)/(r.rolling(60).std()*np.sqrt(20)+1e-12)
fr=np.log(P.shift(-10)/P); ics=[]; dates=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c);dates.append(dt);ns.append(len(z))
a=np.array(ics); print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(np.mean(ns),3),'coverage':round(np.mean(ns)/15,6),'IC':round(a.mean(),6),'ICIR_daily':round(a.mean()/a.std(ddof=1),6),'ICIR_annual':round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit':round(np.mean(a>0),6)})
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-05-25')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))]; print(x,len(z),round(float(z.mean()),6) if len(z) else None,round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),6) if len(z)>1 else None)
