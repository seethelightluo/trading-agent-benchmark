import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=120: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); bench=R.mean(axis=1)
f=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
for s in P.columns:
 beta=R[s].rolling(60,min_periods=40).cov(bench)/bench.rolling(60,min_periods=40).var()
 resid=R[s]-beta*bench
 rev=-resid.rolling(15,min_periods=12).sum()
 vol=resid.rolling(60,min_periods=40).std()
 dispersion=R.rolling(20,min_periods=15).std().mean(axis=1)
 gate=(dispersion/dispersion.rolling(252,min_periods=100).median()).clip(0.5,2.0)
 f[s]=rev/(vol*np.sqrt(15)+1e-8)*gate
f.to_csv('scripts/miner_1_20330609_dispersion_residual_reversal_15d_signal.csv')
fr=R.shift(-10).rolling(10,min_periods=10).sum(); ics=[];dates=[];ns=[];ranks=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c);dates.append(dt);ns.append(len(z));ranks.append(f.loc[dt].rank(pct=True))
a=np.array(ics); S=pd.DataFrame(ranks,index=dates)
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(float(np.mean(ns)),3),'coverage':round(float(np.mean(ns)/15),6),'IC':round(float(a.mean()),6),'ICIR':round(float(a.mean()/a.std(ddof=1)*np.sqrt(252)),6),'hit':round(float(np.mean(a>0)),6),'turnover':round(float(S.diff().abs().mean().mean()),6)})
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-06-08')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))]
 print(x,len(z),round(float(z.mean()),6) if len(z) else None,round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),6) if len(z)>1 else 0 if len(z)>1 else None)
