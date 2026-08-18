import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-07-07'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None: d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); resid=r-r.mean(axis=1); down=resid.where(resid<0,0).rolling(40,min_periods=25).std(); rec=resid.rolling(15,min_periods=10).sum()/(down*np.sqrt(15)+1e-8); rs=resid.rolling(60,min_periods=35).sum(); peak=rs.rolling(60,min_periods=35).max(); dd=rs-peak; fr=np.log(px.shift(-10)/px)
for th in [-.04,-.06,-.08,-.10,-.12]:
 f=(rec*(dd<th)).shift(1); vals=[];ns=[]; sig=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));sig.append(d)
 s=pd.Series(vals,index=sig).dropna(); print(th,len(s),round(s.mean(),6),round(s.mean()/s.std(),6),round(np.mean(s>0),4),round(np.mean(np.array(ns)/15),4))
