import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=get_stock_daily_data(s,4000); d['date']=pd.to_datetime(d.date); fs[s]=d.set_index('date').close.astype(float)
p=pd.concat(fs,axis=1).sort_index(); r=p.pct_change(); med=r.median(axis=1)
# Volatility-scaled residual reversal: fade 20d relative moves, scaled by
# inverse asset volatility, and activate only during unusually high dispersion.
res=r.sub(med,axis=0).rolling(20,min_periods=15).sum()
vol=r.rolling(20,min_periods=15).std()
disp=r.std(axis=1).rolling(20,min_periods=15).mean(); threshold=disp.rolling(252,min_periods=126).quantile(.75)
f=(-res/vol.replace(0,np.nan)).where(disp>threshold).shift(1)
ics=[]; cov=[]; turns=[]; prev=None; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(p.shift(-10)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(v):
   ics.append(v); cov.append(len(z)/15); dates.append(dt)
   q=f.loc[dt].rank(pct=True)
   if prev is not None: turns.append((q-prev).abs().mean())
   prev=q
A=np.array(ics); print('dates',len(A),'instruments',15,'avg_cov',np.mean(cov),'IC',A.mean(),'ICIR',A.mean()/A.std(ddof=1),'hit',(A>0).mean(),'turn',np.mean(turns))
for label,aa in [('all',A),('recent',A[np.array(dates)>=pd.Timestamp('2029-02-25')])]:
 print(label,'n',len(aa),'IC',aa.mean(),'ICIR',aa.mean()/aa.std(ddof=1),'hit',(aa>0).mean())
for i,a in enumerate(np.array_split(A,4),1): print('regime',i,len(a),a.mean(),a.mean()/a.std(ddof=1))
for h in [1,5,10,20]:
 q=[]; yy=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(v): q.append(v)
 q=np.array(q); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300225_volscaled_highdisp20_signal.csv',index=False)
