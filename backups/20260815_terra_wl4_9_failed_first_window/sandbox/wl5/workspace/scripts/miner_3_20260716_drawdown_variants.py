import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U},axis=1).sort_index()
r=p.pct_change(); vol=r.rolling(20).std()
for w in [10,20,40,60,120]:
 for mode in ['raw','voladj']:
  dd=1-p/p.rolling(w).max(); f=dd if mode=='raw' else dd/(vol*np.sqrt(w))
  a=[];ns=[]
  for i in range(w,len(p)-1):
   z=pd.concat([f.iloc[i],p.iloc[i+1]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  a=np.array(a);print(w,mode,'dates',len(a),'avgN',round(np.mean(ns),2),'ic',round(a.mean(),5),'icir',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'recent500',round(a[-500:].mean(),5),round(a[-500:].mean()/a[-500:].std(ddof=1),5))
