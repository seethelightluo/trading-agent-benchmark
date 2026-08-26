import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>260:
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').sort_index()
c=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); ret=c.pct_change()
# Downside-skew weight: assets with frequent/larger negative moves receive stronger rebound score.
down=ret.where(ret<0,0.0)
downvol=down.rolling(60,min_periods=30).std()
totvol=ret.rolling(60,min_periods=30).std()
skew=(downvol/totvol).clip(.25,2.5)
resid=c.pct_change(60).sub(c.pct_change(60).mean(axis=1),axis=0)
sig=(-resid*(0.5+skew.rank(axis=1,pct=True))).shift(1)
rows=[]
for h in [5,10,20,40,60]:
 fwd=c.shift(-h)/c-1; a=[]
 for dt in c.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 x=pd.DataFrame(a,columns=['date','ic','n']); ic=x.ic
 print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/len(D),4),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==20:
  for n,a1,b1 in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
   y=x[(x.date>=a1)&(x.date<=b1)]; print('REG',n,'dates',len(y),'IC',round(y.ic.mean(),6),'ICIR',round(y.ic.mean()/y.ic.std(ddof=1),6),'hit',round((y.ic>0).mean(),4))
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20311016_downside_skew_volume_reversal_signal.csv',index=False)
print('UNIVERSE',len(D),'DATES',len(c),'SIGNAL_ROWS',len(out))
