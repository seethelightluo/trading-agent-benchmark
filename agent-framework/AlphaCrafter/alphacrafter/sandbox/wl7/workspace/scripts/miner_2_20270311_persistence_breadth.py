import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-10')
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 if len(d):
  d=d[d.date<=cut].set_index('date').sort_index(); D[s]=d
print('assets',len(D),[(s,len(d)) for s,d in D.items()])
# candidate: persistence acceleration = recent positive-day breadth minus long breadth, scaled by vol; predicts continuation
rows=[]
for s,d in D.items():
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change()
 f20=r.rolling(20).mean(); f60=r.rolling(60).mean(); v=r.rolling(40).std()
 # stable trend: 20d directional breadth + normalized return, with acceleration of breadth
 fac=(0.6*(r.rolling(20).apply(lambda x: np.mean(x>0),raw=True)-.5)+0.4*(f20/v))* (1+0.5*(r.rolling(20).apply(lambda x: np.mean(x>0),raw=True)-r.rolling(60).apply(lambda x: np.mean(x>0),raw=True)))
 fwd=c.shift(-5)/c-1
 for dt in fac.index:
  if dt<=cut and pd.notna(fac.loc[dt]) and dt in fwd.index and pd.notna(fwd.loc[dt]): rows.append((dt,s,float(fac.loc[dt]),float(fwd.loc[dt])))
x=pd.DataFrame(rows,columns=['date','s','f','y'])
ics=[]; nms=[]
for dt,g in x.groupby('date'):
 if len(g)>=8:
  ic=g.f.corr(g.y,method='spearman'); ics.append(ic); nms.append(len(g))
a=np.array(ics); print('obs',len(a),'avgN',np.mean(nms),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'coverage',len(x)/(len(D)*len(set(x.date))))
for h in [1,5,10,20]:
 z=[]
 for s,d in D.items():
  c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change(); v=r.rolling(40).std(); b=r.rolling(20).apply(lambda q:np.mean(q>0),raw=True); bl=r.rolling(60).apply(lambda q:np.mean(q>0),raw=True)
  f=(.6*(b-.5)+.4*(r.rolling(20).mean()/v))*(1+.5*(b-bl)); y=c.shift(-h)/c-1
  z.append(pd.DataFrame({'f':f,'y':y,'s':s}))
 z=pd.concat(z); q=[]
 for dt,g in z.groupby(level=0):
  g=g.dropna()
  if len(g)>=8:q.append(g.f.corr(g.y,method='spearman'))
 print('decay',h,np.nanmean(q),len(q))
# regime
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-10')]:
 q=[v for dt,v in zip(x.date.unique(),[])]; aa=[]
 for dt,g in x.groupby('date'):
  if str(dt)[:4]>=lo and dt<=pd.Timestamp(hi) and len(g)>=8: aa.append(g.f.corr(g.y,method='spearman'))
 print('regime',lo,hi,np.nanmean(aa),len(aa))
# turnover rank
wide=x.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True); print('turn',wide.diff().abs().mean().mean())
x.to_csv('scripts/miner_2_20270311_persistence_breadth_signal.csv',index=False)
