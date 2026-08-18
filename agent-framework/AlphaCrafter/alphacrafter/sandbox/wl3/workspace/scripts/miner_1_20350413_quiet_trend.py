import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Quiet medium-term trend: 60d return standardized by 60d volatility, rewarded when recent vol is below medium vol.
# All inputs are through date t; forward return is t+10 close / t close - 1.
series={}
for s in U:
    d=get_stock_daily_data(s,days=5000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').close.astype(float)
        series[s]=x
px=pd.concat(series,axis=1).sort_index()
ret=px.pct_change()
vol60=ret.rolling(60,min_periods=45).std()*np.sqrt(252)
vol20=ret.rolling(20,min_periods=15).std()*np.sqrt(252)
r60=px.pct_change(60)
# continuous quiet-trend signal; cap ratio to avoid pathological outliers
f=(r60/vol60)*(vol60/(vol20+1e-12)).clip(0.5,2.0)
f=f.replace([np.inf,-np.inf],np.nan)
fr=px.shift(-10)/px-1
rows=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1,keys=['f','r']).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),z.f.corr(z.r),z))
ics=pd.Series({a:c for a,n,c,z in rows})
print('dates',len(ics),'mean_names',np.mean([n for a,n,c,z in rows]),'coverage',np.mean([n for a,n,c,z in rows])/15)
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0))
for w in [120,252,504,1008]:
 q=ics.tail(w); print('recent',w,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
for h in [1,5,10,20]:
 rr=px.shift(-h)/px-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(vals),len(vals))
# rank turnover on dates with full-ish signal
ranks=f.rank(axis=1,pct=True); ch=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna(); print('turnover_rank_abs_change',ch.mean())
# block means
for i,a in enumerate(np.array_split(ics,4),1): print('block',i,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
