import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}; p=pd.concat(D,axis=1).sort_index(); r=np.log(p).diff()
for name,fast,slow in [('accel_10_40',10,40),('accel_5_20',5,20),('accel_20_80',20,80)]:
 f=(p.pct_change(fast)-p.pct_change(slow))/(r.rolling(slow).std()*np.sqrt(slow)); out={h:[] for h in [1,5,10]}; ns=[]
 for i in range(slow,len(p)-10):
  x=f.iloc[i]
  for h in out:
   z=pd.concat([x,p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8: out[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  if x.notna().sum()>=8: ns.append(x.notna().sum())
 print(name,'dates',len(out[1]),'avgN',np.mean(ns),'cov',np.mean(ns)/15)
 for h,a in out.items():
  a=np.array(a);print(h,round(a.mean(),5),round(a.mean()/a.std(ddof=1),5),round(np.mean(a>0),4))
 print('turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2)
 # correlations with proxies
 print('corr rev5',f.stack().corr((-p.pct_change(5)).stack()),'peer',f.stack().corr(p.pct_change(5).sub(p.pct_change(5).median(axis=1),axis=0).stack()))
