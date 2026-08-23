import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
W=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in W:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').sort_index()
def sig(c):
 # lagged rebound: short recovery from 20d trough, penalized by distance from 120d high
 rebound=c/c.rolling(20).min()-1; dd=c/c.rolling(120).max()-1
 return (rebound/(0.02+(-dd).clip(lower=0))).shift(1)
def panel(h):
 z=[]
 for s,d in D.items():
  c=d.close; z.append(pd.DataFrame({'date':d.index.values,'s':s,'sig':sig(c).values,'fwd':(c.shift(-h)/c-1).values}).dropna())
 return pd.concat(z,ignore_index=True)
def calc(x):
 a=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8:a.append((dt,spearmanr(g.sig,g.fwd).statistic,len(g)))
 return pd.DataFrame(a,columns=['date','ic','n']).set_index('date')
x=panel(10);a=calc(x);print('dates',len(a),'avg_n',a.n.mean(),'assets',len(D),'coverage',len(x)/sum(len(d) for d in D.values()));print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean());print('turnover',x.pivot(index='date',columns='s',values='sig').rank(axis=1,pct=True).diff().abs().mean().mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=a.loc[lo:hi].ic;print('regime',lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [1,5,10,20]:print('decay',h,calc(panel(h)).ic.mean(),len(calc(panel(h))))
x.pivot(index='date',columns='s',values='sig').to_csv('scripts/miner_1_20270706_rebound_signal.csv')
