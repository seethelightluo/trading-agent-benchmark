import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={s:get_stock_daily_data(s,days=5000) for s in U}; p=pd.DataFrame({s:(x.set_index('date')['close'] if x is not None else pd.Series(dtype=float)) for s,x in d.items()}).sort_index().ffill(); r=np.log(p).diff()
# medium momentum with volatility penalty and cross-sectional rank; lag one day
f=r.rolling(40,min_periods=30).sum()/r.rolling(60,min_periods=40).std(); f=f.shift(1); y=np.log(p.shift(-10)/p)
a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
i=pd.DataFrame(a,columns=['date','ic','n']).set_index('date'); print('dates',len(i),'avgN',i.n.mean(),'coverage',i.n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(i.ic.mean(),i.ic.mean()/i.ic.std(),(i.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for n in [60,120,252,756]:q=i.tail(n).ic;print('recent',n,'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
print('period',i.index.min(),i.index.max());f.loc[i.index].to_csv('scripts/miner_2_20320401_medium_momentum_signal.csv')
