import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=get_stock_daily_data(s,5000); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:ld(s) for s in U}).sort_index(); R=P.pct_change(); V=R.rolling(20,min_periods=15).std()
# Test unconditional residual reversal at several medium horizons; lag signal one full day.
for n in [3,5,10]:
 raw=R.rolling(n,min_periods=n).sum(); resid=raw.sub(raw.median(axis=1),axis=0)
 F=(-resid/V).shift(1)
 print('FACTOR',n,'dates',len(P),'assets',len(U),'coverage',F.notna().mean().mean())
 for h in [1,3,5,10]:
  a=[]; ns=[]
  for i in range(len(P)-h):
   z=pd.concat([F.iloc[i],P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1]);
    if np.isfinite(c):a.append(c);ns.append(len(z))
  a=np.array(a); print('h',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 print('turn',np.nanmean(F.rank(axis=1,pct=True).diff().abs().mean(axis=1)))
