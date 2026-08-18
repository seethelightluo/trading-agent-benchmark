import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); f=-P.pct_change(10).shift(1)
ics=[]; counts=[]; prev=None; turns=[]
for dt in f.index:
 x=f.loc[dt]; y=(P.shift(-10)/P-1).loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1]))); counts.append(len(z))
  rr=x.rank(pct=True)
  if prev is not None: turns.append((rr-prev).abs().mean())
  prev=rr
ic=pd.Series(dict(ics)).dropna()
print('ALL dates',len(ic),'avg_names',np.mean(counts),'assets',len(P.columns),'range',P.index.min(),P.index.max())
print('coverage_dates',len(ic)/max(1,len(P)-20),'turnover',np.mean(turns))
for n in [120,252,756,1260,len(ic)]:
 q=ic.tail(n); print('window',n,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(a).dropna(); print('decay',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252))
