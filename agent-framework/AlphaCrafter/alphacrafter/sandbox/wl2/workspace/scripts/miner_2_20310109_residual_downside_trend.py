import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[a]=d[d.date<='2031-01-08'].set_index('date').sort_index()
px=pd.concat({a:d.close for a,d in D.items()},axis=1).sort_index();r=px.pct_change(); m=r.median(axis=1)
# residual trend persistence: 20d cumulative return beyond common median, penalized by downside semideviation
res=r.sub(m,axis=0); down=res.where(res<0,0); risk=down.rolling(30,min_periods=15).std(); f=(res.rolling(20,min_periods=20).sum()/risk.replace(0,np.nan)).shift(1)
fr=px.pct_change().shift(-1);vals=[];ns=[];ds=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
x=np.array(vals);print('dates',len(x),'avgN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/np.std(x,ddof=1),6),'hit',round(np.mean(x>0),4),'instruments',len(D))
for h in [1,3,5,10]:
 ff=px.pct_change(h).shift(-h);v=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'IC',round(np.mean(v),6),'n',len(v))
