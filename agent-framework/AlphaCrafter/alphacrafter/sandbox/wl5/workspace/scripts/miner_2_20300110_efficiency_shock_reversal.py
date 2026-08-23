import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,4000) for s in U}
C=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=C.pct_change(); ret20=C/C.shift(20)-1; path=r.abs().rolling(20).sum(); eff=ret20/path.replace(0,np.nan)
f=-eff*(1+r.rolling(5).std()/r.rolling(30).std().replace(0,np.nan))
for h in [3,5,10,20]:
 vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],C.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(vals); print('H',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('coverage',f.notna().mean().mean(),'avgN',f.notna().sum(axis=1).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300110_efficiency_shock_reversal_signal.csv',index=False)
