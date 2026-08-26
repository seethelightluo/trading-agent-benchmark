import numpy as np,pandas as pd, os
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-10-31'); D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').drop_duplicates('date'); D[s]=x.set_index('date')
P=pd.concat({s:x.close.astype(float) for s,x in D.items()},axis=1).sort_index(); R=P.pct_change(); M=R.mean(axis=1)
rows=[]
for s,x in D.items():
 c=x.close.astype(float); r=R[s].reindex(c.index); mm=M.reindex(c.index)
 beta=r.rolling(60,min_periods=40).cov(mm)/mm.rolling(60,min_periods=40).var().replace(0,np.nan)
 # Fade a medium shock ending five sessions ago, avoiding overlap with today's close.
 shock=r.rolling(10,min_periods=10).sum()-beta*mm.rolling(10,min_periods=10).sum()
 f=-shock.shift(5)/r.rolling(40,min_periods=25).std().replace(0,np.nan)
 for h in [1,5,10,20]:
  rows.append(pd.DataFrame({'date':c.index,'symbol':s,'factor':f,'fwd':c.shift(-h)/c-1,'h':h}))
a=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
for h,g0 in a.groupby('h'):
 out=[]
 for d,g in g0.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>=3 and g.fwd.nunique()>=3: out.append((d,g.factor.corr(g.fwd),len(g)))
 z=pd.DataFrame(out,columns=['date','ic','n']); v=z.ic
 print('H',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(v.mean(),8),'ICIR',round(v.mean()/v.std(ddof=1),8),'hit',round((v>0).mean(),4))
 if len(z)>=3:
  n=len(z); print(' thirds',*[round(v.iloc[i*n//3:(i+1)*n//3].mean(),6) for i in range(3)])
if len(D):
 a[a.h==10][['date','symbol','factor']].to_csv('scripts/miner_1_20321101_skip5_residual10_signal.csv',index=False)
print('universe',len(D),'rows',len(a),'calendar_dates',len(P.index),'coverage',round(len(a[a.h==10])/(len(D)*len(P.index)),4))
