import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
C=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index(); R=C.pct_change()
# medium-horizon cross-sectional residual reversal, skip two sessions, risk scaled
m=R.rolling(10).sum(); resid=m.sub(m.median(axis=1),axis=0); v=R.rolling(40).std()*np.sqrt(40)
f=(-resid/v).shift(2).replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [1,5,10,20]:
 F=C.shift(-h)/C-1; z=[]
 for t in f.index:
  q=pd.concat([f.loc[t],F.loc[t]],axis=1).dropna()
  if len(q)>=8:z.append((t,q.iloc[:,0].corr(q.iloc[:,1]),len(q)))
 z=pd.DataFrame(z,columns=['date','ic','n']).set_index('date'); mu=z.ic.mean(); sd=z.ic.std(ddof=1)
 print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,len(z),z.n.mean(),mu,mu/sd*np.sqrt(252), (z.ic>0).mean()))
 if h==10:
  for a in np.array_split(z,3):print('third',len(a),a[:,1].mean())
print('cutoff',C.index.max(),'assets',len(U),'coverage',f.notna().sum().sum()/f.size,'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_2_20330110_medium_residual_reversal_signal.csv',index=False)
