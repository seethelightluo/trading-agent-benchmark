import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# trend acceleration: recent 20d return minus prior 40d return, scaled by lagged 30d vol
D={s:get_stock_daily_data(s,days=5000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None})
px.index=pd.to_datetime(px.index); px=px.sort_index(); r=px.pct_change()
# lagged signal at t uses through t, evaluated t+1; construct at dates
ret20=px/px.shift(20)-1; ret60=px/px.shift(60)-1
# acceleration relative to old 40d segment: r20 - (r60-r20)/2
sig=(ret20-(ret60-ret20)/2)/(r.rolling(30).std().shift(1)*np.sqrt(20))
# ensure signal lag one day
fwd=r.shift(-1)
rows=[]
for h in [1,5,10]:
  ic=[]
  for dt in sig.index:
    x=sig.loc[dt]; y=(px.shift(-h)/px-1).loc[dt]
    z=pd.concat([x,y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)>=8: ic.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
  q=pd.DataFrame(ic,columns=['date','ic','n']).set_index('date')
  mean=q.ic.mean(); sd=q.ic.std(ddof=1); ir=mean/sd*np.sqrt(252) if sd else np.nan
  print('H',h,'dates',len(q),'avg_n',q.n.mean(),'IC',mean,'ICIR',ir,'hit', (q.ic>0).mean())
  if h==1:
   # regimes
   n=len(q); print('regimes',*[q.ic.iloc[a:b].mean() for a,b in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
# turnover and coverage based rank changes
valid=sig.notna().sum(axis=1)/len(U); ranks=sig.rank(axis=1,pct=True)
to=(ranks-ranks.shift(1)).abs().mean(axis=1)
print('rows',len(px),'range',px.index.min(),px.index.max(),'coverage',valid.mean(),'turnover',to.mean())
# artifacts
sig.to_csv('scripts/miner_3_20310519_trend_acceleration_signal.csv')
