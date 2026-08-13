import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=get_stock_daily_data(s,5000);d.date=pd.to_datetime(d.date);return d.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:ld(s) for s in U}).sort_index();R=P.pct_change();V=R.rolling(20,min_periods=15).std(); raw=R.rolling(3,min_periods=3).sum(); res=raw.sub(raw.median(axis=1),axis=0)
disp=R.rolling(5,min_periods=5).std().mean(axis=1)
for qv in [.50,.60,.70,.80]:
 gate=disp.shift(1)>disp.shift(1).rolling(252,min_periods=100).quantile(qv);F=(-res/V).shift(1).where(gate)
 a=[];ns=[]
 for i in range(len(P)-1):
  z=pd.concat([F.iloc[i],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c):a.append(c);ns.append(len(z))
 a=np.array(a);print('q',qv,'active',gate.sum(),'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean(F.notna().mean().where(gate)))
