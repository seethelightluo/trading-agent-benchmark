import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 gap=np.log(d.open/d.close.shift(1))
 # efficiency of overnight drift: persistent gap direction relative to its own dispersion
 f=gap.rolling(20,min_periods=15).mean()/(gap.rolling(20,min_periods=15).std()+1e-12)
 D[a]=pd.DataFrame({'f':f,'r1':np.log(d.close.shift(-1)/d.close),'r5':np.log(d.close.shift(-5)/d.close),'r10':np.log(d.close.shift(-10)/d.close),'r20':np.log(d.close.shift(-20)/d.close)})
allx=pd.concat(D,names=['asset','date']).reset_index().pivot(index='date',columns='asset')
def calc(col):
 vals=[];ns=[]
 for dt,row in allx['f'].iterrows():
  z=pd.concat([row,allx[col].loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 v=np.array(vals);return len(v),np.mean(ns),v.mean(),v.mean()/v.std(ddof=1),np.mean(v>0)
for h in [1,5,10,20]:print('H',h,calc('r'+str(h)))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030')]:
 vals=[]
 for dt,row in allx['f'].iterrows():
  if lo<=dt.strftime('%Y')<=hi:
   z=pd.concat([row,allx['r1'].loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 v=np.array(vals); print('REGIME',lo,hi,len(v),v.mean(),v.mean()/v.std(ddof=1))
print('coverage',round(allx['f'].notna().mean().mean(),4),'dates',len(allx),'meanN overall',allx['f'].notna().sum(axis=1).mean())
