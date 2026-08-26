import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>220:D[s]=d.set_index('date').sort_index()
c=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); r=c.pct_change()
# Recovery-quality reversal: favor assets with deep 120d drawdowns only when short trend has stabilized.
# Drawdown supplies contrarian value; 20d return and volatility penalize falling knives.
peak=c.rolling(120,min_periods=60).max(); dd=c/peak-1
vol=r.rolling(40,min_periods=20).std()*np.sqrt(252)
trend=c.pct_change(20)/(vol+1e-12)
# positive stabilization gate, lagged to prevent lookahead
sig=(dd/(vol+1e-12) * (1/(1+np.exp(-2*trend)))).shift(1)
rows=[]
for h in [5,10,20,40,60]:
 fwd=c.shift(-h)/c-1; a=[]
 for dt in c.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(a,columns=['date','ic','n'])
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 if h==60:
  for n,a1,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
   x=q[(q.date>=a1)&(q.date<=b)]; print('REG',n,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(ddof=1),6) if len(x)>1 else np.nan,round((x.ic>0).mean(),4) if len(x) else np.nan)
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20310626_recovery_reversal_signal.csv',index=False)
print('UNIVERSE',len(D),'DATES',len(c),'SIGNAL_ROWS',len(out))
