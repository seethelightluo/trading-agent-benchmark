import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={s:get_stock_daily_data(s,days=4000) for s in U}
c=pd.concat({s:d.set_index('date')['close'] for s,d in raw.items()},axis=1).sort_index().ffill(); r=c.pct_change()
# short-horizon reversal scaled by recent volatility, with a volatility-shock amplifier; lagged inputs
base=-r.shift(1).rolling(3,min_periods=3).sum()/r.shift(1).rolling(20,min_periods=15).std()
shock=r.shift(1).rolling(5,min_periods=4).std()/r.shift(1).rolling(60,min_periods=40).std()
sig=(base*shock.clip(0.5,3)).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10,20]:
 y=c.shift(-h)/c-1; a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna();print('H',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'n',len(a))
print('coverage',sig.notna().sum(axis=1).mean()/15,'dates',len(sig))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20331212_shock_reversal_signal.csv',index=False);print('artifact',len(out))
