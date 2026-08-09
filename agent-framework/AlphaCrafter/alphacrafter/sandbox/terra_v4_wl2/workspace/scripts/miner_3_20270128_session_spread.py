import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in syms:
 f='../persistent/stock_data/'+s+'.csv';
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 D[s]=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
rows=[]
for s,x in D.items():
 prev=x.close.shift(1); overnight=x.open/prev-1; intra=x.close/x.open-1
 # robust 3d average of intraday minus overnight: captures session-specific relative strength
 fac=(intra-overnight).rolling(3,min_periods=3).mean()
 fr=x.close.shift(-1)/x.close-1
 rows.append(pd.DataFrame({'date':x.index,'sym':s,'f':fac,'fr':fr}).dropna())
a=pd.concat(rows); p=a.pivot(index='date',columns='sym',values='f'); r=a.pivot(index='date',columns='sym',values='fr'); ic=[]; ns=[]
for dt in p.index:
 z=pd.concat([p.loc[dt],r.loc[dt]],axis=1).dropna()
 if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
ic=np.array(ic); print('dates',len(ic),'avg_names',np.mean(ns),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0),'coverage',sum(ns)/(len(ic)*15))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2026-07-16','2027-01-27')]:
 q=[]
 for dt in p.loc[lo:hi].index:
  z=pd.concat([p.loc[dt],r.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
p.stack if False else None
p.stack().rename('signal').reset_index().to_csv('../persistent/factor_signals_miner_3_20270128_session_spread3.csv',index=False)
