import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<='2031-01-08'].set_index('date').sort_index(); D[a]=d
px=pd.concat({a:d.close for a,d in D.items()},axis=1).sort_index(); hi=pd.concat({a:d.high for a,d in D.items()},axis=1).reindex(px.index); lo=pd.concat({a:d.low for a,d in D.items()},axis=1).reindex(px.index); vol=pd.concat({a:d.volume for a,d in D.items()},axis=1).reindex(px.index)
r=px.pct_change(); rng=(hi-lo).replace(0,np.nan)
# Range-location pressure: persistent close location, scaled by abnormal volume, with cross-sectional demeaning.
clv=(2*px-hi-lo)/rng
vrel=vol/vol.rolling(20,min_periods=10).median()
base=clv.rolling(5,min_periods=5).mean()*np.log1p(vrel.clip(lower=0))
f=base.sub(base.median(axis=1),axis=0).shift(1)
fr=px.pct_change().shift(-1)
vals=[]; ns=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
x=np.asarray(vals); print('dates',len(x),'avgN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/len(A),4),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/np.nanstd(x,ddof=1),6),'hit',round(np.mean(x>0),4),'rows',len(px),'instruments',len(D))
for lo,hi_d in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2031')]:
 y=x[(np.array(dates)>=pd.Timestamp(lo+'-01-01'))&(np.array(dates)<=pd.Timestamp(hi_d+'-12-31'))]
 print(lo+'-'+hi_d,'n',len(y),'IC',round(np.nanmean(y),6),'ICIR',round(np.nanmean(y)/np.nanstd(y,ddof=1),6) if len(y)>1 else None)
for h in [1,3,5,10]:
 ff=px.pct_change(h).shift(-h); vv=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vv.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',round(float(np.nanmean(vv)),6),'n',len(vv))
