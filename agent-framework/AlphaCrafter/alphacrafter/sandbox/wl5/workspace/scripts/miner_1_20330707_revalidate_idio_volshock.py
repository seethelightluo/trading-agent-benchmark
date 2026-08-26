import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1)
f=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
for s in P.columns:
 b=R[s].rolling(40,min_periods=30).cov(m)/m.rolling(40,min_periods=30).var()
 e=R[s]-b*m; shock=e.rolling(10,min_periods=8).sum(); iv=e.rolling(20,min_periods=15).std()
 gate=iv>iv.rolling(126,min_periods=80).quantile(.75); f[s]=-shock*(1+.5*gate.astype(float))
f.stack().rename('factor_value').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330707_idio_volshock_signal.csv',index=False)
fr=R.shift(-10).rolling(10,min_periods=10).sum(); ics=[];ds=[];ns=[]; ranks=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c);ds.append(dt);ns.append(len(z));ranks.append(f.loc[dt].rank(pct=True))
a=np.array(ics); S=pd.DataFrame(ranks,index=ds)
print({'dates':len(a),'start':str(ds[0].date()),'end':str(ds[-1].date()),'mean_n':round(np.mean(ns),3),'coverage':round(np.mean(ns)/15,6),'IC':round(a.mean(),6),'ICIR':round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit':round(np.mean(a>0),6),'turnover':round(S.diff().abs().mean().mean(),6)})
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-07-06')]:
 z=a[(np.array(ds)>=pd.Timestamp(x))&(np.array(ds)<=pd.Timestamp(y))]
 print(x,len(z),round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1)*np.sqrt(252),6) if len(z)>1 else None)
