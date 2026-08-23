import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}
P=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index().ffill()
r=P.pct_change(); r5=P/P.shift(5)-1; vol20=r.rolling(20).std()*np.sqrt(252)
# Cross-asset residual short-term reversal, normalized by own recent volatility.
# Signal is lagged one session; higher values indicate expected forward outperformance.
resid=r5-r5.median(axis=1).values[:,None]
F=(-(resid/vol20.replace(0,np.nan))).shift(1)
print('data',P.index.min().date(),P.index.max().date(),'dates',len(P),'assets',len(D))
for h in [5,10,20,40]:
 q=[]; ns=[]
 fr=P.shift(-h)/P-1
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(x): q.append(x);ns.append(len(z))
 a=np.array(q); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(a>0),4))
print('turnover',round(F.rank(pct=True).diff().abs().mean(axis=1).mean(),6),'overall_coverage',round(F.notna().sum(axis=1).mean()/15,6))
