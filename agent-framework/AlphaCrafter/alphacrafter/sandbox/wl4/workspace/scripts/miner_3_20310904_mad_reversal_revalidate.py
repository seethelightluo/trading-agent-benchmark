import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cutoff=pd.Timestamp('2031-09-03');p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'];p[s]=d[d.index<=cutoff]
px=pd.DataFrame(p).sort_index();r=px.pct_change();res=r.rolling(15,min_periods=10).sum();res=res.sub(res.median(axis=1),axis=0);med=r.rolling(40,min_periods=20).median();mad=(r-med).abs().rolling(40,min_periods=20).median()*1.4826;vol=r.rolling(40,min_periods=20).std();scale=.7*mad+.3*vol
f=(-res/(scale+1e-8)).shift(1);disp=r.rolling(20,min_periods=10).std().mean(axis=1);gate=(disp>=disp.rolling(252,min_periods=60).median()).astype(float);f=f.mul(gate,axis=0)
fr=px.shift(-10)/px-1;v=[];ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
x=pd.Series(v);print('dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean());print('coverage',f.notna().sum().sum()/px.notna().sum().sum(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'instruments',len(U),'dates_total',len(px))
for n in [365,730]:
 y=x.iloc[-n:];print('recent',n,'IC',y.mean(),'ICIR',y.mean()/y.std(ddof=1),'dates',len(y))
