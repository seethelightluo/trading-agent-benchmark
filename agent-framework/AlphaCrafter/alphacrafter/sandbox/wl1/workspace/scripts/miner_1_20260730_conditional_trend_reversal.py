import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index(); p.columns=U; p=p.ffill(); r=p.pct_change()
# Conditional trend: 20d momentum only when trailing 5d cross-asset breadth is positive; otherwise use 5d reversal.
breadth=(r>0).mean(axis=1).rolling(5).mean(); mom=p.pct_change(20); rev=-p.pct_change(5); f=mom.where(breadth>=.5,rev)
for H in [1,5,10]:
 ic=[]; ns=[]
 for i in range(len(p)-H):
  q=pd.concat([f.iloc[i],(p.iloc[i+H]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: ic.append(spearmanr(q.iloc[:,0],q.y).statistic); ns.append(len(q))
 x=np.array(ic); print('H',H,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
print('active bullish',int((breadth>=.5).sum()),'assets',len(U),'range',p.index.min().date(),p.index.max().date())
