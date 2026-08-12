import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def read(a, macro=False):
 f=('../persistent/index_data/' if macro else '../persistent/stock_data/')+a+'.csv'
 return pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
px=pd.concat({a:read(a) for a in A},axis=1).sort_index(); vix=read('VIX',True).reindex(px.index).ffill()
r=np.log(px).diff(); vr=np.log(vix).diff()
# Momentum is downweighted after abrupt volatility spikes, reducing crowded risk-on exposure
shock=vr.rolling(5,min_periods=3).mean().clip(-.2,.2)
f=(np.log(px/px.shift(20)).mul((1-shock/0.2),axis=0)).shift(1)
y=r.shift(-1); dates=[];ics=[];ns=[];tos=[]; prev=None
for dt in px.index:
 if dt>pd.Timestamp('2028-05-03'): break
 x=f.loc[dt]; z=pd.concat([x,y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; ics.append(ic);ns.append(len(z));dates.append(dt)
  q=x.rank(pct=True)
  if prev is not None: tos.append(np.abs(q-prev).dropna().mean())
  prev=q
arr=np.array(ics);print('idea=vix_shock_conditioned_momentum');print('dates',len(arr),'avgN',np.mean(ns),'IC',arr.mean(),'ICIR',arr.mean()/arr.std(ddof=1),'hit',np.mean(arr>0),'turnover',np.mean(tos),'coverage',np.mean(ns)/15,'period',dates[0].date(),dates[-1].date())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2028-05-03')]:
 b=arr[(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<=pd.Timestamp(hi))]; print(lo,hi,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
for h in [5,10]:
 yy=np.log(px.shift(-h)/px); aa=[]
 for dt in dates:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',np.mean(aa),'ICIR',np.mean(aa)/np.std(aa,ddof=1),'n',len(aa))
