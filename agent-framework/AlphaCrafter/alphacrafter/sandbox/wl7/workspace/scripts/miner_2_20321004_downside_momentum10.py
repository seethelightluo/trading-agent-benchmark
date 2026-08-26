import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={s:get_stock_daily_data(s,days=5000) for s in U}
close=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index()
r=np.log(close).diff(); ret10=r.rolling(10).sum()
# Downside deviation over 40 sessions; zero/positive returns do not contribute
neg=r.where(r<0,0.0); down=(neg.pow(2).rolling(40).mean()).pow(.5)*np.sqrt(40)
f=(ret10/down).shift(1).replace([np.inf,-np.inf],np.nan)
fr={h:close.shift(-h)/close-1 for h in [1,5,10,20]}
print('cutoff',close.index.max().date(),'assets',len(U),'dates',len(close))
print('coverage %.4f'%(f.notna().sum().sum()/f.size))
ranks=f.rank(axis=1,pct=True); print('turnover10 %.4f'%((ranks-ranks.shift(10)).abs().mean(axis=1).mean()))
for h in [1,5,10,20]:
 vals=[]
 for dt in f.index:
  x=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(x)>=8: vals.append((dt,x.iloc[:,0].corr(x.iloc[:,1]),len(x)))
 z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); mean=z.ic.mean(); sd=z.ic.std(ddof=1)
 print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,len(z),z.n.mean(),mean,mean/sd if sd else np.nan,(z.ic>0).mean()))
 if h==10:
  for a in np.array_split(z,3): print('third',len(a), 'IC %.6f'%a[:,1].mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20321004_downside_momentum10_signal.csv',index=False)
