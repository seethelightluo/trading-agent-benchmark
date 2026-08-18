import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close
P=pd.DataFrame(D).sort_index(); r=P.pct_change(); vol=r.rolling(30).std()
# Intermediate-horizon trend acceleration: recent 20d return relative to prior 40d return, risk normalized.
F=((P.pct_change(20)-P.pct_change(60)/3).div(vol*np.sqrt(20))).shift(1)
F.to_csv('scripts/miner_2_20341124_acceleration_signal.csv',index_label='date')
print('assets',len(P.columns),'dates',len(P),'coverage',F.notna().mean().mean())
for h in [1,3,5,10,20]:
 a=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(a); ir=a.mean()/a.std(ddof=1)*np.sqrt(len(a))
 print('horizon',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',ir,'hit',np.mean(a>0))
 if h==10:
  for n in [120,252,504,756,1260]:
   q=a[-n:];print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
# mean cross-sectional rank turnover
R=F.rank(axis=1,pct=True); print('turnover',np.nanmean(np.abs(R-R.shift(1)).mean(axis=1)))
