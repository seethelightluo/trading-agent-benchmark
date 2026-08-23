import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-03-22')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change();
# Residual momentum: asset 10d return demeaned by contemporaneous equal-weight cross-asset return, lagged one day.
raw=r.rolling(10,min_periods=10).sum(); market=raw.mean(axis=1); sig=(raw.sub(market,axis=0)).shift(1)
for h in [1,3,5,10]:
 f=px.shift(-h)/px-1; a=[]; ds=[]; ns=[]
 for dt in px.index:
  g=pd.DataFrame({'x':sig.loc[dt],'y':f.loc[dt]}).dropna()
  if len(g)>=8 and g.x.nunique()>1:
   a.append(spearmanr(g.x,g.y).statistic); ds.append(dt); ns.append(len(g))
 a=np.array(a); recent=a[np.array(ds)>=END-pd.Timedelta(days=180)]
 print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'recentIC',round(recent.mean(),6),'recentICIR',round(recent.mean()/recent.std(ddof=1),6))
 if h==1:
  for name,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026','2026-01-01','2026-12-31'),('2027','2027-01-01','2027-12-31'),('2028','2028-01-01','2028-03-22')]:
   z=np.array([v for v,d in zip(a,ds) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)])
   print(name,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('coverage',float(np.mean(~sig.isna().sum(axis=1)/15)))
