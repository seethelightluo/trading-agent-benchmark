import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=3200)
 if d is not None: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Smooth risk-adjusted trend: medium momentum penalized by both total risk and recent volatility shock.
vol20=r.rolling(20,min_periods=15).std()*np.sqrt(20)
vol60=r.rolling(60,min_periods=40).std()*np.sqrt(60)
shock=(r.rolling(5,min_periods=4).std()*np.sqrt(5))/(vol60/np.sqrt(3)+1e-12)
mom=np.log(p).diff(20)
f=(mom/(vol20+1e-12)/(1+0.6*shock)).sub((mom/(vol20+1e-12)/(1+0.6*shock)).median(axis=1),axis=0).shift(1)
print('rows',len(p),'assets',len(D),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=np.log(p).shift(-h)-np.log(p); a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q);ns.append(len(z))
 a=np.array(a); print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,mask in [('2027+',f.index>='2027-01-01'),('2028YTD',f.index>='2028-01-01'),('recent',f.index>='2028-05-01')]:
 y=np.log(p).shift(-10)-np.log(p); a=[]
 for dt in f.index[mask]:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q)
 a=np.array(a); print(label,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
print('turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),6))
f.to_csv('scripts/miner_3_20281116_smooth_risk_trend_signal.csv')
