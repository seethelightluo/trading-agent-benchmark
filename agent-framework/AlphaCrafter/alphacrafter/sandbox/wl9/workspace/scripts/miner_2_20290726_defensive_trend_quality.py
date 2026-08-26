import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s,n=4000):
 d=get_stock_daily_data(s,n)
 if d is None or len(d)==0:return pd.Series(dtype=float)
 return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
px=pd.DataFrame({s:get(s) for s in U}).sort_index(); ret=px.pct_change()
# Durable trend quality: medium-term return rewarded, penalized by downside volatility and drawdown.
r60=px.pct_change(60)
down=ret.where(ret<0).rolling(40,min_periods=20).std()*np.sqrt(40)
dd=(px/px.rolling(160,min_periods=80).max()-1).clip(-1,0)
sig=r60/(down+1e-6)*(1+0.35*dd)

def calc(h, start=None):
 vals=[]; ds=[]; ns=[]
 for d in sig.index:
  if start is not None and d<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[d],(px.shift(-h)/px-1).loc[d]],axis=1).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
   if np.isfinite(v): vals.append(v);ds.append(d);ns.append(len(z))
 a=np.array(vals); ds=np.array(ds,dtype='datetime64[ns]')
 return a,ds,ns
base_dates=len(sig.dropna(how='all'))
for h in [5,10,20,40]:
 a,ds,ns=calc(h)
 print('h',h,'dates',len(a),'mean_n',round(np.mean(ns),2),'coverage',round(len(a)/base_dates,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10:
  for label,cut in [('online','2026-07-16'),('recent252','2028-06-28'),('2029','2029-01-01'),('2027','2027-01-01')]:
   q=a[ds>=np.datetime64(cut)]; print(label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
# turnover of daily cross-sectional ranks (Spearman adjacent)
a=[]
for d in sig.index[1:]:
 x,y=sig.shift(1).loc[d],sig.loc[d]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: a.append(1-z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
print('turnover_proxy',np.nanmean(a),'instruments',len(U))
