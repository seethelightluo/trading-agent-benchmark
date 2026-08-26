import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={s:get_stock_daily_data(s,days=5000) for s in U}; close=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index(); r=np.log(close).diff()
base=-(r.rolling(10).sum().sub(r.rolling(10).sum().median(axis=1),axis=0))/(r.rolling(40).std()*np.sqrt(40)); f=base.rolling(5,min_periods=3).mean().shift(1).replace([np.inf,-np.inf],np.nan)
print('cutoff',close.index.max().date(),'dates',len(close),'assets',len(U),'coverage %.4f'%(f.notna().sum().sum()/f.size)); rank=f.rank(axis=1,pct=True); print('turnover10 %.4f'%((rank-rank.shift(10)).abs().mean(axis=1).mean()))
for h in [1,5,10,20]:
 fr=close.shift(-h)/close-1; vals=[]
 for dt in f.index:
  x=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(x)>=8: vals.append((dt,x.iloc[:,0].corr(x.iloc[:,1]),len(x)))
 z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); m=z.ic.mean(); sd=z.ic.std(ddof=1); print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,len(z),z.n.mean(),m,m/sd,(z.ic>0).mean()))
 if h==10:
  for start,end in [(0,len(z)//3),(len(z)//3,2*len(z)//3),(2*len(z)//3,len(z))]: print('third dates=%d IC=%+.6f'%(end-start,z.iloc[start:end]['ic'].mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_2_20321018_smoothed_residual_reversal_signal.csv',index=False)
