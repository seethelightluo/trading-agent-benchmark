import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s, days=5000) for s in U}
px={s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
sig=-(P.shift(1)/P.shift(6)-1)/(r.rolling(20).std().shift(1)*np.sqrt(5))
fwd=P.shift(-10)/P-1
rows=[]; dates=[]; ns=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append(x[ok].corr(y[ok],method='spearman')); dates.append(dt); ns.append(ok.sum())
ic=pd.Series(rows,index=dates).dropna()
print('dates',len(ic),'avgN',np.mean(ns),'coverage',sig.notna().mean().mean())
print('IC10',ic.mean(),'ICIR_daily',ic.mean()/ic.std(),'hit', (ic>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [1,5,20]:
 fy=P.shift(-h)/P-1; z=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fy.loc[dt].notna()
  if ok.sum()>=8:z.append(sig.loc[dt,ok].corr(fy.loc[dt,ok],method='spearman'))
 print('decay',h,np.nanmean(z))
for n in [365,750,1260]: print('recent',n,ic.tail(n).mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20351108_short_shock_reversal_signal.csv',index=False)
ic.rename('ic').to_csv('scripts/miner_1_20351108_short_shock_reversal_ic.csv')
