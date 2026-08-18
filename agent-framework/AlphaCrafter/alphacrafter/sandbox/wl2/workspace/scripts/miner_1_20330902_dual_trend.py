import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); v=r.rolling(40).std().shift(1)*np.sqrt(40)
# medium trend with short-term confirmation; fully lagged and normalized
f=(p.pct_change(40).shift(1)/v + p.pct_change(10).shift(1)/(r.rolling(20).std().shift(1)*np.sqrt(20)))/2
for h in [1,3,5,10]:
 y=p.pct_change(h).shift(-h); a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print(h,len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0))
print('coverage',f.notna().mean().mean(),'assets',len(D),'dates',len(p)); f.to_csv('scripts/miner_1_20330902_dual_trend_signal.csv')
