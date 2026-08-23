import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);xs[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(xs).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1)
macro=pd.read_csv('../persistent/index_data/DXY.csv');macro.date=pd.to_datetime(macro.date)
dxy=macro.set_index('date').close.astype(float).reindex(p.index).ffill().pct_change()
rows=[]; art=[]
for i in range(120,len(p)-10):
 dt=p.index[i]; rr=r.iloc[:i+1]; mm=m.iloc[:i+1]
 # medium horizon residual reversal, beta estimated causally from last 120 observations
 b=rr.iloc[-120:].apply(lambda x:x.cov(mm.iloc[-120:])/(mm.iloc[-120:].var()+1e-12))
 resid=rr.iloc[-20:].sum()-b*mm.iloc[-20:].sum()
 vol=rr.iloc[-60:].std(); db=dxy.iloc[:i+1].iloc[-60:]
 w=rr.iloc[-60:].loc[db.index]; bv=w.apply(lambda x:x.cov(db)/(db.var()+1e-12))
 # damp assets with unusually high DXY beta; bounded and interpretable
 z=(bv-bv.median())/(bv.std()+1e-9); gate=(1-0.20*z).clip(.65,1.35)
 sig=-resid/(vol+1e-9)*gate
 y=p.iloc[i+10]/p.iloc[i]-1
 q=pd.DataFrame({'f':sig,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8:
  ic=q.f.corr(q.y);rows.append((dt,float(ic),len(q)))
  for s,v in sig.items():
   if s in q.index: art.append({'date':dt,'symbol':s,'signal':float(v),'ic':float(ic)})
q=np.array([x[1] for x in rows]);print('dates',len(q),'mean_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15)
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2032-09-15')]:
 v=np.array([x[1] for x in rows if pd.Timestamp(a)<=x[0]<=pd.Timestamp(b)]);print(a,len(v),v.mean() if len(v) else np.nan,v.mean()/v.std(ddof=1) if len(v)>1 else np.nan)
pd.DataFrame(art).to_csv('scripts/miner_1_20320916_medium_dxy_residual_signal.csv',index=False)
