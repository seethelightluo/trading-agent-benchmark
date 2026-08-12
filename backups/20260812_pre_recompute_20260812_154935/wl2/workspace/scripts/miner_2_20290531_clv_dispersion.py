import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<='2029-05-30'].set_index('date').sort_index(); D[a]=d
px=pd.concat({a:d.close for a,d in D.items()},axis=1).sort_index(); hi=pd.concat({a:d.high for a,d in D.items()},axis=1).reindex(px.index); lo=pd.concat({a:d.low for a,d in D.items()},axis=1).reindex(px.index); r=px.pct_change()
clv=((px-lo)/(hi-lo).replace(0,np.nan)-.5); disp=r.std(axis=1).rolling(20,min_periods=15).mean(); gate=disp>disp.rolling(252,min_periods=120).median()
for lb in [3,5,10,20]:
 f=clv.rolling(lb,min_periods=lb).mean().where(gate,np.nan).shift(1); fr=px.pct_change().shift(-1); vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(vals); print(lb,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/np.nanstd(x,ddof=1),6),'hit',round(np.mean(x>0),4))
print('instruments',len(D),'rows',len(px))
