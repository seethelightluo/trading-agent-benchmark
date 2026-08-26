import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
C=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index(); R=C.pct_change()
short=R.rolling(5).sum(); resid=short.sub(short.median(axis=1),axis=0)
vol=R.rolling(30,min_periods=8).std().replace(0,np.nan); disp=R.std(axis=1).rolling(5).mean()
threshold=disp.rolling(120,min_periods=60).quantile(.65); gate=(disp>threshold).astype(float).replace(0,np.nan)
f=(-resid/vol*gate).shift(2).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10,20]:
 F=C.shift(-h)/C-1; vals=[]
 for t in f.index:
  q=pd.concat([f.loc[t],F.loc[t]],axis=1).dropna()
  if len(q)>=8: vals.append((t,q.iloc[:,0].corr(q.iloc[:,1]),len(q)))
 z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); mu=z.ic.mean(); sd=z.ic.std(ddof=1)
 print(f'H{h} dates={len(z)} avgN={z.n.mean():.2f} IC={mu:+.6f} ICIR={mu/sd*np.sqrt(252):+.6f} hit={(z.ic>0).mean():.4f}')
 if h==10:
  for i,a in enumerate(np.array_split(z.index,3)): print(f'third{i+1} dates={len(a)} IC={z.loc[a,"ic"].mean():+.6f}')
print('cutoff',C.index.max(),'assets',len(U),'coverage',f.notna().sum().sum()/f.size,'active_dates',gate.notna().sum())
print('turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_2_20330124_shock_recovery_signal.csv',index=False)
