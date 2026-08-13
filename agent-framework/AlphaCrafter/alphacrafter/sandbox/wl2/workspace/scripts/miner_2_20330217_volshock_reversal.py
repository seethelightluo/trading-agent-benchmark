import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=get_stock_daily_data(s,5000); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:ld(s) for s in U}).sort_index(); R=P.pct_change(); v=R.rolling(20,min_periods=15).std(); shock=v/v.rolling(120,min_periods=60).median(); short=R.rolling(5,min_periods=5).sum(); med=short.median(axis=1); F=(-(short.sub(med,axis=0))/v).shift(1).where(shock.shift(1)>1.35); F.to_csv('scripts/miner_2_20330217_volshock_reversal_signal.csv')
for h in [1,3,5,10]:
 a=[];ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c):a.append(c);ns.append(len(z))
 a=np.array(a);print('horizon',h,'dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('total_dates',len(P),'assets',len(U),'active',int((shock.shift(1)>1.35).any(axis=1).sum()),'coverage_active',np.nanmean(F.notna().mean(axis=1).where(F.notna().any(axis=1))))
for lo,hi in [('2026','2029-12-31'),('2030','2033-12-31')]:
 a=[]
 for i in np.where((F.index>=lo)&(F.index<=hi))[0]:
  if i+1>=len(P):continue
  z=pd.concat([F.iloc[i],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c):a.append(c)
 a=np.array(a);print('regime',lo,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
