import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; data={}
for s in watch:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); data[s]=d.set_index('date').sort_index()
def signal(c):
 low=c.rolling(60).min(); dd=c/low-1
 sl=c.rolling(60).apply(lambda z: len(z)-1-np.argmin(z),raw=True)
 return ((c/c.shift(20)-1)/(1+sl/20.0) * (1+dd).clip(.5,2)).shift(1)
def panel(h):
 rows=[]
 for s,d in data.items():
  c=d.close; rows.append(pd.DataFrame({'date':d.index.values,'s':s,'sig':signal(c).values,'fwd':(c.shift(-h)/c-1).values}).dropna())
 return pd.concat(rows,ignore_index=True)
def calc(xx):
 vals=[]
 for dt,g in xx.groupby('date'):
  if len(g)>=8: vals.append((dt,spearmanr(g.sig,g.fwd).statistic,len(g)))
 return pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
x=panel(10); a=calc(x)
print('dates',len(a),'avg_n',a.n.mean(),'assets',len(data),'coverage',len(x)/sum(len(d) for d in data.values()))
print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
r=x.pivot(index='date',columns='s',values='sig').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=a.loc[lo:hi].ic; print('regime',lo,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [1,5,10,20]:
 q=calc(panel(h)); print('decay',h,q.ic.mean(),len(q))
x.pivot(index='date',columns='s',values='sig').to_csv('scripts/miner_1_20270706_recovery_velocity_signal.csv')
