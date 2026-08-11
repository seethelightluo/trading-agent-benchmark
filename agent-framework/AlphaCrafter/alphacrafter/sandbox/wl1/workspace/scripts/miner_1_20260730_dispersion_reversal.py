import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index()
r=p.pct_change(); w=3; f=pd.DataFrame(index=p.index,columns=U,dtype=float)
# Dispersion-conditioned short-term reversal. Dispersion threshold is trailing-only median.
for i in range(61,len(p)):
  x=r.iloc[i-w+1:i+1]; disp=x.iloc[-1].std()
  hist=r.iloc[i-60:i].rolling(w).std().mean(axis=1).dropna()
  if len(hist) and disp>hist.median(): f.iloc[i]=-r.iloc[i-w:i+1].sum(axis=0)
for h in [1,5,10]:
 ic=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q))
 x=np.array(ic); print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
print('active',f.notna().any(axis=1).sum(),'assets',len(U))
