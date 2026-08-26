import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>180:D[s]=d.set_index('date').sort_index()
c=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); r=c.pct_change()
ret=c.pct_change(40)
down=r.clip(upper=0).rolling(60).std()*np.sqrt(252)
# Contrarian rebound: penalize assets with persistent downside volatility; lagged signal.
sig=(-ret/(down+1e-12)).shift(1)
fwd=c.shift(-40)/c-1
rows=[]
for dt in c.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n'])
print('DATES',len(q),'AVG_N',round(q.n.mean(),2),'COVERAGE',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'HIT',round((q.ic>0).mean(),4))
for h in [5,10,20,40,60]:
 f=c.shift(-h)/c-1; a=[]
 for dt in c.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=pd.Series(a);print('DECAY',h,round(a.mean(),6),round(a.mean()/a.std(),6),len(a))
for n,a,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
 x=q[(q.date>=a)&(q.date<=b)];print('REG',n,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(),6) if len(x)>1 else np.nan,round((x.ic>0).mean(),4) if len(x) else np.nan)
print('TURNOVER',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'UNIVERSE',len(D),'CALENDAR_DATES',len(c))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20310515_asymmetric_rebound_40d_signal.csv',index=False)
