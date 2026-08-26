import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>180:D[s]=d.set_index('date').sort_index()
c=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); r=c.pct_change()
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
# Contrarian recent return, strengthened when short volatility is compressed versus its long baseline.
compression=(v20/(v60+1e-12)).clip(0.25,3.0)
sig=(-(c.pct_change(20)/(v20*np.sqrt(20)+1e-12))*(1+0.5*(1-compression))).shift(1)
rows=[]
for h in [5,10,20,40,60]:
 fwd=c.shift(-h)/c-1; a=[]
 for dt in c.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(a,columns=['date','ic','n'])
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if h==20:
  for n,a1,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
   x=q[(q.date>=a1)&(q.date<=b)]; print('REG',n,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(),6) if len(x)>1 else np.nan,round((x.ic>0).mean(),4) if len(x) else np.nan)
  turn=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(); print('TURN',round(turn,6))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20310626_volcompression_reversal_signal.csv',index=False)
print('UNIVERSE',len(D),'DATES',len(c),'SIGNAL_ROWS',len(out))
