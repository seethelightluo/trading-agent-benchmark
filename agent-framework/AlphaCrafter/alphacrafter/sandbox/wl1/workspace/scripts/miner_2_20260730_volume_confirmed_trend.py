import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
P={};V={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index(); P[s]=d.close; V[s]=d.volume
p=pd.DataFrame(P);v=pd.DataFrame(V); ret20=p/p.shift(20)-1
vr=np.log((v.rolling(5,min_periods=3).mean()+1e-12)/(v.rolling(60,min_periods=30).mean()+1e-12))
# Volume-confirmed trend: momentum strengthened when recent volume exceeds its baseline.
f=(ret20*vr.clip(-2,2)).rank(axis=1,pct=True)
for h in [1,5,10]:
 a=[];n=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:a.append(spearmanr(q.iloc[:,0],q.y).statistic);n.append(len(q))
 x=np.array(a);print('horizon',h,'dates',len(x),'avgN',round(np.mean(n),2),'coverage',round(np.mean(n)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
print('rows',len(p),'assets',len(U))
