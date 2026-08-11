import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
P={};V={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index()
 P[s]=d.close; V[s]=d.volume
p=pd.DataFrame(P); v=pd.DataFrame(V); r=p.pct_change()
# Volume-confirmed short reversal: recent loss is more likely to mean-revert when volume is unusually high.
ret3=p/p.shift(3)-1
vr=np.log((v.rolling(3,min_periods=2).mean()+1e-12)/(v.rolling(20,min_periods=10).mean()+1e-12))
f=(-ret3*vr.clip(-2,2)).rank(axis=1,pct=True)
for h in [1,5,10]:
 vals=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q))
 x=np.array(vals); print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
vals=[]
for i in range(len(p)-10):
 q=pd.concat([f.iloc[i],(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(q)>=8: vals.append((f.index[i],spearmanr(q.iloc[:,0],q.y).statistic))
z=pd.Series(dict(vals));print('annual10d',{int(y):round(z[z.index.year==y].mean(),6) for y in sorted(z.index.year.unique())})
turn=[]
for i in range(1,len(f)):
 q=f.iloc[i].dropna().index.intersection(f.iloc[i-1].dropna().index)
 if len(q)>=8:turn.append(np.abs(f.iloc[i][q]-f.iloc[i-1][q]).mean())
print('rank_turnover',round(np.mean(turn),6),'rows',len(p),'assets',len(U))
