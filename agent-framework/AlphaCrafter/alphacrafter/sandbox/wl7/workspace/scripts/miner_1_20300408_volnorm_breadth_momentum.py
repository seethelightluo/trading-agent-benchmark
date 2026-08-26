import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,4200); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index(); r=p.pct_change()
# Volatility-normalized intermediate momentum, with a cross-asset breadth gate.
# Breadth is lagged and only changes signal magnitude, not asset ranking in positive regimes.
ret20=p.pct_change(20); vol60=r.rolling(60,min_periods=40).std()*np.sqrt(20); breadth=(ret20>0).mean(axis=1)
gate=(0.5+0.5*breadth).clip(.25,1.0)
f=(ret20/vol60).mul(gate,axis=0).shift(1)
ics=[]; cov=[]; turns=[]; prev=None
for dt in f.index:
 z=pd.concat([f.loc[dt],(p.shift(-10)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(v):
   ics.append(v); cov.append(len(z)/15)
   q=f.loc[dt].rank(pct=True)
   if prev is not None: turns.append((q-prev).abs().mean())
   prev=q
A=np.array(ics)
print('dates',len(A),'instruments',15,'avg_cov',np.mean(cov),'coverage',np.mean(cov),'IC',A.mean(),'ICIR',A.mean()/A.std(ddof=1),'hit',(A>0).mean(),'turn',np.mean(turns))
for i,a in enumerate(np.array_split(A,4),1): print('regime',i,len(a),a.mean(),a.mean()/a.std(ddof=1))
for h in [1,5,10,20,40]:
 yy=p.shift(-h)/p-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(v): q.append(v)
 q=np.array(q); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('cutoff',p.index.max())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300408_volnorm_breadth_momentum_signal.csv',index=False)
