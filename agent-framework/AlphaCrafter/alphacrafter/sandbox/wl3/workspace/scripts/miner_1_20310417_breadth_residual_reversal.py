import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=p.pct_change(); rr=r.rolling(3).sum(); vol=r.rolling(20).std(); med=rr.median(axis=1); f=-rr.sub(med,axis=0)/vol
breadth=(r>0).sum(axis=1)/r.notna().sum(axis=1)
for cut in [.15,.20,.25,.30,.35]:
 sig=f.where((breadth<cut)|(breadth>1-cut)); vals=[];ns=[]; fr=r.rolling(5).sum()
 for i in range(40,len(p)-5):
  z=pd.concat([sig.iloc[i-1],fr.iloc[i+4]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 a=np.array(vals); print('cut',cut,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
