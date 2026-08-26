import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>180: D[s]=d.set_index('date').sort_index()
close=pd.DataFrame({s:d.close for s,d in D.items()}); ret=close.pct_change()
dur=pd.DataFrame(index=close.index,columns=close.columns,dtype=float)
for s in close:
 x=close[s].values; out=np.full(len(x),np.nan); last=-1
 for i,v in enumerate(x):
  if np.isfinite(v) and i>=59 and v >= np.nanmax(x[max(0,i-59):i+1])*(1-1e-10): last=i
  if last>=0: out[i]=i-last
 dur[s]=out
vol=ret.rolling(20).std()*np.sqrt(252)
base=dur/(vol+1e-12)
breadth=(ret.rolling(20).sum()>0).mean(axis=1)
# Stress weighting: duration reversal receives greater weight when fewer assets have positive 20d returns.
sig=(base.mul(1.0+(1.0-breadth),axis=0)).shift(1)
for h in [5,10,20,40,60]:
 vals=[]
 for i in range(len(close.index)-h):
  z=pd.concat([sig.iloc[i],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append((close.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n'])
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/len(U),4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if h==20:
  for name,a,b in [('2024-2026','2024-01-01','2026-12-31'),('2027-2029','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
   x=q[(q.date>=a)&(q.date<=b)]; print('REG',name,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(),6) if len(x)>1 else np.nan,round((x.ic>0).mean(),4) if len(x) else np.nan)
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20310403_breadth_conditioned_duration_signal.csv',index=False)
print('UNIVERSE',len(D),'DATES',len(close))
