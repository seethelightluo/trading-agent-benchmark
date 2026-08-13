import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=get_stock_daily_data(s,5000); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:ld(s) for s in U}).sort_index(); R=P.pct_change(); V=R.rolling(20,min_periods=15).std(); raw=R.rolling(3,min_periods=3).sum(); res=raw.sub(raw.median(axis=1),axis=0)
disp=R.rolling(5,min_periods=5).std().mean(axis=1); past=disp.shift(1); q=past.rolling(252,min_periods=100).quantile(.75); active=past>q
intensity=((past/q)-1).clip(lower=0,upper=1); F=(-res/V).shift(1).mul(intensity.shift(1),axis=0).where(active.shift(1)); F.index.name='date'; F.to_csv('scripts/miner_2_20330303_dispersion75_capped_signal.csv')
print('total_dates',len(P),'assets',len(U),'active_dates',int(active.sum()),'coverage',float(F.notna().mean().mean()))
for h in [1,3,5,10]:
 a=[];ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1])
   if np.isfinite(c):a.append(c);ns.append(len(z))
 a=np.asarray(a); print('horizon',h,'dates',len(a),'avg_instruments',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
rank=F.rank(axis=1,pct=True); print('turnover',float(np.nanmean(((rank-rank.shift()).abs().mean(axis=1)).where(active))))
for lo,hi in [('2026','2029-12-31'),('2030','2033-12-31')]:
 a=[]
 for i in np.where((F.index>=lo)&(F.index<=hi))[0]:
  if i+1>=len(P):continue
  z=pd.concat([F.iloc[i],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c):a.append(c)
 a=np.asarray(a); print('regime',lo,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
