import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=4000)
    if x is not None and len(x): D[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float)
px=pd.concat(D,axis=1).sort_index(); rets=px.pct_change()
r60=px.pct_change(60); down=rets.clip(upper=0).rolling(60).std()*np.sqrt(252)
f=(r60/(down+1e-8)).shift(1); fr=px.shift(-10)/px-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',a.index.min(),a.index.max(),'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('mean_ic %.6f icir %.6f hit %.4f turnover %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for w in [120,260,520,780,1200]:
 q=a.tail(w); print(w,'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
for h in [5,10,20,40]:
 q=px.shift(-h)/px-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('horizon',h,'ic',np.nanmean(rr),'icir',np.nanmean(rr)/np.nanstd(rr))
