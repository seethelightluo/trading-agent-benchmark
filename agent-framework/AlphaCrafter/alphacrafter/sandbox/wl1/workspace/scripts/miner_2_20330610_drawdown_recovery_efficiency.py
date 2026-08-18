import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs=[]
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 q=d.copy(); q.date=pd.to_datetime(q.date); xs.append(q.drop_duplicates('date').set_index('date').close.rename(s))
p=pd.concat(xs,axis=1).sort_index().ffill(); r=np.log(p).diff(); bench=r.mean(axis=1); res=r.sub(bench,axis=0)
down=res.where(res<0).rolling(40,min_periods=25).std(); raw=res.rolling(20,min_periods=20).sum()/(down*np.sqrt(20))
dd=(p/p.rolling(60,min_periods=60).max()-1).clip(-1,0)
sig=raw.mul(1-dd.abs()).shift(1)
rows=[]
for i,dt in enumerate(p.index[:-21]):
 z=pd.concat([sig.loc[dt],(np.log(p).iloc[i+10]-np.log(p).iloc[i])],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); print('range',p.index.min(),p.index.max(),'assets',p.shape[1],'dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/(len(o)*15)); print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean()))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=o.loc[a:b].ic; print(a+'-'+b,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [5,20]:
 rr=[]
 for i,dt in enumerate(p.index[:-h-1]):
  z=pd.concat([sig.loc[dt],(np.log(p).iloc[i+h]-np.log(p).iloc[i])],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('h',h,'IC',np.nanmean(rr),'n',len(rr))
print('turnover',sig.rank(pct=True).diff().abs().mean().mean()); sig.stack().rename('signal').reset_index().to_csv('scripts/miner_2_20330610_drawdown_recovery_efficiency_signal.csv',index=False)
