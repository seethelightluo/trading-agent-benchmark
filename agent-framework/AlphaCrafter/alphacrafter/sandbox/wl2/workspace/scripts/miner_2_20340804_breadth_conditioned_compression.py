import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=[s for s in get_account_dict().get('watch_list',[]) if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<200:d=get_index_daily_data(s,5000)
 if d is not None:px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill(); r=np.log(p).diff()
# Breadth-conditioned volatility compression: reward compressed assets only when
# the lagged cross-asset 20d breadth is positive; otherwise invert toward defensive assets.
comp=-(r.rolling(10).std()/r.rolling(60).std())
breadth=(r.rolling(20).sum().gt(0).mean(axis=1)-0.5).shift(1)
fac=(comp.mul(np.where(breadth.values[:,None]>=0,1.0,-1.0),axis=0)).shift(1)
rows=[]
for h in [1,5,10,20,40]:
 fw=np.log(p).shift(-h)-np.log(p); ic=[]; ns=[]
 for d in fac.index:
  z=pd.concat([fac.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=pd.Series(ic).dropna(); print('H',h,'DATES',len(x),'AVG_N',round(np.mean(ns),2),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1)*np.sqrt(252):.6f}','HIT',f'{(x>0).mean():.4f}')
# latest regime breakdown at 40d
fw=np.log(p).shift(-40)-np.log(p); rows=[]
for d in fac.index:
 z=pd.concat([fac.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
o=pd.DataFrame(rows,columns=['date','ic']).set_index('date')
for a,b in [('2026-07-16','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2034-08-03')]:
 q=o.loc[a:b,'ic'];print('REG',a,b,'DATES',len(q),'IC',f'{q.mean():.6f}','ICIR',f'{q.mean()/q.std(ddof=1)*np.sqrt(252):.6f}')
fac.to_csv('scripts/miner_2_20340804_breadth_conditioned_compression_signal.csv')
