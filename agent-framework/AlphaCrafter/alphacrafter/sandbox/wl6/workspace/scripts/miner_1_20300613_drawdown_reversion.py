import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); D[s]=d.sort_values('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); r10=p/p.shift(10)-1
peak=p.rolling(60).max(); dd=p/peak-1; vol=r.rolling(20).std()
# Revert recent losses, with larger signal for established drawdowns, risk scaled
f=(-r10)*(-dd).clip(lower=0)/vol.replace(0,np.nan)
for h in [5,10,20]:
 vals=[]; ns=[]
 for i in range(len(p)-h):
  a=f.iloc[i]; b=p.iloc[i+h]/p.iloc[i]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic); ns.append(ok.sum())
 z=np.array(vals); print('h',h,'dates',len(z),'avg_n',np.mean(ns),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1),'hit',np.mean(z>0))
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
