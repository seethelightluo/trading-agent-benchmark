import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; TODAY='2030-02-25'; px={}
for s in U:
 try:d=get_index_daily_data(s,days=4000)
 except Exception:d=None
 if d is None:
  try:d=get_stock_daily_data(s,days=4000)
  except Exception:d=None
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]; r=np.log(P).diff(); v=r.rolling(20,min_periods=15).std();
d20=P/P.rolling(20,min_periods=20).max()-1; d60=P/P.rolling(60,min_periods=60).max()-1
sig=((d20-d60)/v).shift(1); rows=[]
for t in sig.index:
 z=pd.concat([sig.loc[t],np.log(P.shift(-10).loc[t]/P.loc[t])],axis=1).dropna()
 if len(z)>=8: rows.append((t,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=out.ic
print('dates',len(q),'avg_n',out.n.mean(),'coverage',sig.notna().mean().mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-02-25')]:
 x=q.loc[lo:hi];print('regime',lo, len(x),x.mean(),x.mean()/x.std())
for h in [1,5,10,20,40]:
 a=[]
 for t in sig.index:
  z=pd.concat([sig.loc[t],np.log(P.shift(-h).loc[t]/P.loc[t])],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'ic',np.nanmean(a),'dates',len(a))
out.to_csv('scripts/miner_3_20300225_drawdown_acceleration_reval_ic.csv');sig.to_csv('scripts/miner_3_20300225_drawdown_acceleration_reval_signal.csv')
