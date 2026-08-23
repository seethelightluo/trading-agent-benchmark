import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
P=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index().ffill()
r=P.pct_change(); r5=P/P.shift(5)-1; r20=P/P.shift(20)-1
# residual reversal removes the contemporaneous common cross-asset move; lagged signal
f=-(r5-r5.median(axis=1).values[:,None])*(1+0.25*np.sign(r20))
F=pd.DataFrame(f,index=P.index,columns=P.columns).shift(1)
print('data',P.index.min().date(),P.index.max().date(),'dates',len(P),'assets',len(D))
for h in [5,10,20,40]:
 q=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],(P.shift(-h)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(x):q.append(x);ns.append(len(z))
 a=np.array(q);print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'hit',np.mean(a>0))
print('turnover',F.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',F.notna().sum(axis=1).mean()/15)
