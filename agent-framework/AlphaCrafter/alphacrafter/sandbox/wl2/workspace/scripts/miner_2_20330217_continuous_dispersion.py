import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=get_stock_daily_data(s,5000); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:ld(s) for s in U}).sort_index(); R=P.pct_change(); V=R.rolling(20,min_periods=15).std(); raw=R.rolling(3,min_periods=3).sum(); res=raw.sub(raw.median(axis=1),axis=0)
disp=R.rolling(5,min_periods=5).std().mean(axis=1); hist=disp.shift(1).rolling(252,min_periods=100); pct=disp.shift(1).rolling(252,min_periods=100).apply(lambda x: (x[-1]>=x).mean(),raw=True)
# continuous tail intensity: zero below median, rising linearly to 1 at 90th percentile
q50=hist.quantile(.50); q90=hist.quantile(.90); intensity=((pct-.50)/.40).clip(0,1)
F=((-res/V).shift(1).mul(intensity,axis=0)); F.index.name='date'; F.to_csv('scripts/miner_2_20330217_continuous_dispersion_signal.csv')
for h in [1,3,5,10]:
 a=[];ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c):a.append(c);ns.append(len(z))
 a=np.asarray(a); print('horizon',h,'dates',len(a),'avg_instruments',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
rank=F.rank(axis=1,pct=True); print('total_dates',len(P),'assets',len(U),'mean_coverage',np.nanmean(F.notna().mean(axis=1)),'turnover',np.nanmean((rank-rank.shift()).abs().mean(axis=1)))
for lo,hi in [('2026','2029-12-31'),('2030','2033-12-31')]:
 a=[]
 for i in np.where((F.index>=lo)&(F.index<=hi))[0]:
  if i+1>=len(P):continue
  z=pd.concat([F.iloc[i],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c):a.append(c)
 a=np.asarray(a);print('regime',lo,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
for name,sub in [('low',pct<.5),('mid',(pct>=.5)&(pct<.8)),('high',pct>=.8)]:
 a=[]
 for i in np.where(sub.fillna(False))[0]:
  if i+1>=len(P):continue
  z=pd.concat([F.iloc[i],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c):a.append(c)
 a=np.asarray(a);print('bucket',name,'dates',len(a),'IC',a.mean() if len(a) else np.nan,'ICIR',a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
