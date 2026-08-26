import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def F(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,4200)
   if d is not None and len(d): return d.set_index(pd.to_datetime(d.date))
  except: pass
P={s:F(s) for s in U};P={s:d for s,d in P.items() if d is not None};p=pd.DataFrame({s:d.close for s,d in P.items()}).sort_index();r=p.pct_change()
# medium-horizon residual reversal with downside-risk scaling and breadth conditioning
r20=r.rolling(20,min_periods=15).sum(); resid=r20.sub(r20.median(axis=1),axis=0)
down=r.where(r<0).rolling(40,min_periods=25).std(); fac=(-resid/(down*np.sqrt(20)+1e-12)); breadth=(r20<0).mean(axis=1);fac=fac.mul((.6+.4*breadth).clip(.6,1),axis=0).shift(1)
f=p.shift(-20).div(p)-1;a=[];ns=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:
  c=z.iloc[:,0].corr(z.iloc[:,1]);
  if np.isfinite(c):a.append(c);ns.append(len(z))
a=np.array(a);print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for n in [252,756,1260]:q=a[-n:];print('recent',n,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
top=fac.rank(axis=1,pct=True)>=.8;ch=[]
for i in range(1,len(top)):
 z=top.iloc[i-1].notna()&top.iloc[i].notna()
 if z.sum()>=8:ch.append((top.iloc[i-1][z]!=top.iloc[i][z]).mean())
print('coverage',fac.notna().mean().mean(),'turnover',np.mean(ch),'instruments',len(p.columns));fac.to_csv('scripts/miner_1_20350723_downside_residual_reversal20_signal.csv',index_label='date')
