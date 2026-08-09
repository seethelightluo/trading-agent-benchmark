import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:get_stock_daily_data(a,days=2400).set_index('date').close.astype(float) for a in A}
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change()
v=get_index_daily_data('VIX',days=2400).set_index('date').close.astype(float).reindex(p.index).ffill()
assetvol=r.rolling(20,min_periods=15).std(); f=(-p.pct_change(5)).div(assetvol*np.sqrt(5))
vr=(v/v.rolling(252,min_periods=60).median()).clip(.5,2.5); f=f.mul((.75+.5*vr).clip(.5,2),axis=0)
print('assets',len(A),'dates',len(p))
for h in [1,5,10]:
 y=r.shift(-1).rolling(h).sum() # t+1...t+h (rolling alignment needs shift after)
 y=r.shift(-h).rolling(h).sum().shift(-(h-1))
 # simpler explicit
 y=sum(r.shift(-k) for k in range(1,h+1))
 ics=[];ns=[];tr=[];prev=None
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: ics.append(q.f.corr(q.y));ns.append(len(q))
  z=f.iloc[i].rank(pct=True)
  if prev is not None: tr.append(np.nanmean(abs(z-prev)))
  prev=z
 x=np.asarray(ics);x=x[np.isfinite(x)]
 print('horizon',h,'dates',len(x),'avg_names',np.mean(ns),'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1),'hit',np.mean(x>0),'coverage',np.mean(ns)/15,'turnover',np.nanmean(tr))
# daily recent/annual
rows=[]; y=r.shift(-1)
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1: rows.append((p.index[i],q.f.corr(q.y)))
z=pd.DataFrame(rows,columns=['date','ic']).set_index('date')
for w in [60,120,252]:
 x=z.ic.tail(w);print('recent',w,len(x),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0))
for yr,g in z.groupby(z.index.year):print('year',yr,len(g),g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1))
print('corr_simple_reversal',pd.concat([f.stack(),(-p.pct_change(5)).stack()],axis=1).dropna().corr().iloc[0,1])
