import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
W=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in W:
 p='../persistent/stock_data/'+s+'.csv'
 x=pd.read_csv(p,parse_dates=['date']).set_index('date')
 d[s]=x.close
px=pd.DataFrame(d).sort_index().loc[:'2033-03-30']; r=px.pct_change()
# Drawdown recovery: trend strength is rebound from the 60-day trough, tempered by
# whether the latest 10-day return confirms recovery. Lag one completed day.
low=px.rolling(60,min_periods=40).min()
recovery=(px/low-1).clip(-1,3)
confirm=r.rolling(10,min_periods=8).sum()
sig=(recovery*confirm/(r.rolling(20,min_periods=15).std()+1e-12)).shift(1)
print('candidate drawdown_recovery_60_10_vol; dates',len(px),'assets',len(px.columns))
print('coverage',round(sig.notna().mean().mean(),4),'mean_valid',round(sig.notna().sum(axis=1).replace(0,np.nan).mean(),2),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; vals=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,lo,hi in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-03-30')]:
 vals=[];fw=px.shift(-1)/px-1
 for dt in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(vals); print('REG1',label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else 'NA','hit',round((a>0).mean(),4))
print('AUDIT_REQUIRED: if efficacy gates pass, compute max absolute Spearman correlation against every admitted factor signal before persistence.')
