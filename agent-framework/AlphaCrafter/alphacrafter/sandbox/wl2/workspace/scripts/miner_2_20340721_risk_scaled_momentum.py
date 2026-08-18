import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[])
U=[x for x in U if x not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)<200: d=get_index_daily_data(s,days=5000)
 if d is not None: px[s]=d.set_index('date')['close'].astype(float)
prices=pd.DataFrame(px).sort_index().ffill(); r=np.log(prices).diff()
# candidate medium momentum risk scaled; values at t use thru t-1, forward starts t+1
fac=(np.log(prices).diff(60)/r.rolling(20).std()).shift(1)
for h in [1,5,10,20,40]:
 fwd=np.log(prices).shift(-h)-np.log(prices)
 vals=[]; ninst=[]
 for dt in fac.index:
  a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ninst.append(len(z))
 x=pd.Series(vals).dropna(); ic=x.mean(); ir=ic/x.std(ddof=1)*np.sqrt(252); hit=(x>0).mean()
 print(h,'dates',len(x),'avgN',np.mean(ninst),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,hit))
# annual-ish regimes for 40
h=40; fwd=np.log(prices).shift(-h)-np.log(prices); rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for a,b in [('2026-07-16','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2034-07-20')]:
 q=out.loc[a:b,'ic']; print('REG',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),(out.loc[a:b,'n']).mean())
print('coverage',fac.notna().sum(axis=1).mean()/len(U))
# signal artifact
fac.to_csv('scripts/miner_2_20340721_risk_scaled_momentum_signal.csv')
