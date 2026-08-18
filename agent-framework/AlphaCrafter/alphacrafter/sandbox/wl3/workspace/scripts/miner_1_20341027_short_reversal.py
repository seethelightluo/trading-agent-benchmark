import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# short-horizon cross-sectional reversal, lagged one completed session
px={}
for s in U:
 d=get_stock_daily_data(s, days=6000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# factor is negative trailing 5d return, known at t; forward 10d return t+1..t+10
f=-P.pct_change(5).shift(1)
fwd=P.shift(-10)/P-1
ics=[]; turnovers=[]; counts=[]
prev=None
for dt in f.index:
 x=f.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1]))); counts.append(len(z))
  ranks=x.rank(pct=True)
  if prev is not None:
   turnovers.append(np.mean((ranks-prev).abs()))
  prev=ranks
ic=pd.Series(dict(ics)).dropna()
for n in [120,252,756,1260]:
 q=ic.tail(n); print('window',n,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan,'hit', (q>0).mean())
print('ALL dates',len(ic),'avg_names',np.mean(counts),'coverage',len(ic)/(len(P)-15 if len(P)>15 else len(P)),'turnover',np.mean(turnovers))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(vals).dropna(); print('decay',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252))
print('range',P.index.min(),P.index.max(),'assets',len(P.columns))
