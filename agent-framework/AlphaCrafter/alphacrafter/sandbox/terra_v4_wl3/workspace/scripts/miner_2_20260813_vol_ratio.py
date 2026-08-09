import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.loc[:'2026-07-15']
 r=np.log(d).diff(); f=-(r.rolling(20,min_periods=15).std()/r.rolling(60,min_periods=40).std())
 fw=d.pct_change(1).shift(-1)
 rows.append(pd.DataFrame({'date':d.index,'symbol':s,'f':f.values,'fw':fw.values}))
x=pd.concat(rows).dropna(); z=x.pivot(index='date',columns='symbol',values='f'); y=x.pivot(index='date',columns='symbol',values='fw')
for h in [1]:
 vals=[];ns=[]
 for date in z.index:
  a=pd.concat([z.loc[date],y.loc[date]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>1: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ns.append(len(a))
 vals=np.array(vals);print('dates',len(vals),'meanN',np.mean(ns),'IC',vals.mean(),'ICIR',vals.mean()/vals.std(ddof=1),'hit',(vals>0).mean(),'coverage',len(x)/sum(len(pd.read_csv('../persistent/stock_data/'+s+'.csv')) for s in U))
for h in [5,10]:
 fw=[]
 for s in U:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.loc[:'2026-07-15'];fw.append(d.pct_change(h).shift(-h).rename(s))
 yy=pd.concat(fw,axis=1); vals=[];ns=[]
 for date in z.index:
  a=pd.concat([z.loc[date],yy.loc[date]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>1:vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ns.append(len(a))
 vals=np.array(vals);print('h',h,'dates',len(vals),'meanN',np.mean(ns),'IC',vals.mean(),'ICIR',vals.mean()/vals.std(ddof=1))
z.to_csv('scripts/miner_2_20260813_vol_ratio_signal.csv')
