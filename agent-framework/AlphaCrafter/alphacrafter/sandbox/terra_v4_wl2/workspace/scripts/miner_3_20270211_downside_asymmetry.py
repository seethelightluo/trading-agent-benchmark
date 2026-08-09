import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in syms:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 D[s]=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
rows=[]
for s,x in D.items():
 ret=x.close.pct_change()
 # downside asymmetry: assets with less downside than upside movement are favored
 dn=ret.clip(upper=0).rolling(20,min_periods=20).std()
 up=ret.clip(lower=0).rolling(20,min_periods=20).std()
 fac=np.log((up+1e-8)/(dn+1e-8))
 fr=x.close.shift(-1)/x.close-1
 rows.append(pd.DataFrame({'date':x.index,'sym':s,'f':fac,'fr':fr}).dropna())
a=pd.concat(rows);p=a.pivot(index='date',columns='sym',values='f');r=a.pivot(index='date',columns='sym',values='fr')
def calc(pp,rr):
 z=[];ns=[]
 for dt in pp.index:
  q=pd.concat([pp.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z);return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0),sum(ns)/(len(z)*15),z
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2026-07-16','2027-02-10')]:
 print(lo,hi,calc(p.loc[lo:hi],r.loc[lo:hi])[:-1])
print('FULL',calc(p,r)[:-1])
p.stack if False else None
p.stack().rename('signal').reset_index().to_csv('../persistent/factor_signals_miner_3_20270211_downside_asymmetry.csv',index=False)
