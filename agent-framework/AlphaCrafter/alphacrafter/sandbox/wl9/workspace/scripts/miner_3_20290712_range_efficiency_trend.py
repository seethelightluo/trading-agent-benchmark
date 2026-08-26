import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:return pd.DataFrame()
 d=d.copy();d.date=pd.to_datetime(d.date);return d.set_index('date').sort_index()
ds={s:load(s) for s in U}
cl=pd.DataFrame({s:d.close.astype(float) for s,d in ds.items()}); hi=pd.DataFrame({s:d.high.astype(float) for s,d in ds.items()}); lo=pd.DataFrame({s:d.low.astype(float) for s,d in ds.items()})
r=cl.pct_change(); r20=cl.pct_change(20)
# Trend quality: directional return relative to realized absolute path, penalizing noisy paths.
path=r.abs().rolling(20,min_periods=15).sum(); sig=r20/(path+1e-8)
fwd=cl.shift(-10)/cl-1
rows=[]; dates=[]; ns=[]
for dt in sig.index:
 q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(q)>=8:
  x=q.iloc[:,0].rank(); y=q.iloc[:,1].rank(); v=x.corr(y)
  if np.isfinite(v):rows.append(v);dates.append(dt);ns.append(len(q))
ic=np.asarray(rows); dates=np.asarray(dates)
print('factor=20d path-efficiency trend; dates',len(ic),'mean_n',np.mean(ns),'coverage',len(ic)/len(sig.dropna(how='all')))
print('IC %.8f ICIR %.8f hit %.6f'%(ic.mean(),ic.mean()/ic.std(ddof=1),np.mean(ic>0)))
for label,start in [('warmup',pd.Timestamp('2020-01-01')),('online',pd.Timestamp('2026-07-16')),('recent',pd.Timestamp('2028-07-12')),('latest',pd.Timestamp('2029-01-01'))]:
 a=ic[dates>=start];print(label,'n',len(a),'IC %.8f ICIR %.8f hit %.6f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for h in [1,5,10,20,40]:
 yy=cl.shift(-h)/cl-1; aa=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=q.iloc[:,0].rank().corr(q.iloc[:,1].rank())
   if np.isfinite(v):aa.append(v)
 aa=np.asarray(aa);print('h',h,'n',len(aa),'IC %.8f ICIR %.8f'%(aa.mean(),aa.mean()/aa.std(ddof=1)))
