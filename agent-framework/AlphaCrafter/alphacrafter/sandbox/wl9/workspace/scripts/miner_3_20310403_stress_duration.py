import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>180:D[s]=d.set_index('date').sort_index()
close=pd.DataFrame({s:d.close for s,d in D.items()}); r=close.pct_change(); dur=pd.DataFrame(index=close.index,columns=close.columns,dtype=float)
for s in close:
 x=close[s].values;o=np.full(len(x),np.nan);last=-1
 for i,v in enumerate(x):
  if np.isfinite(v) and i>=59 and v>=np.nanmax(x[i-59:i+1])*(1-1e-10):last=i
  if last>=0:o[i]=i-last
 dur[s]=o
v=r.rolling(20).std()*np.sqrt(252); base=dur/(v+1e-12); breadth=(r.rolling(20).sum()>0).mean(axis=1)
# nonlinear stress regime: retain signal in broad weakness, compress it in broad strength
mult=pd.Series(np.where(breadth<0.4,1.5,np.where(breadth>0.67,0.35,0.8)),index=close.index)
sig=base.mul(mult,axis=0).shift(1)
for h in [5,10,20,40,60]:
 a=[]
 for i in range(len(close)-h):
  z=pd.concat([sig.iloc[i],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append((close.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(a,columns=['date','ic','n']);print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if h==20:
  for n,a1,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
   x=q[(q.date>=a1)&(q.date<=b)];print('REG',n,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(),6) if len(x)>1 else np.nan,round((x.ic>0).mean(),4) if len(x) else np.nan)
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20310403_stress_duration_signal.csv',index=False);print('UNIVERSE',len(D),'DATES',len(close))
