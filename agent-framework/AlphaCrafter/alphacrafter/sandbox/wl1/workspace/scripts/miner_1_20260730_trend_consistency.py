import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index()
r=p.pct_change()
# Trend-consistency factor: medium momentum is retained only when short and medium trends agree.
# Sign and magnitude are computed from data available at decision date.
m5=p/p.shift(5)-1; m20=p/p.shift(20)-1
f=m20.where(np.sign(m5)==np.sign(m20),0.0)
for h in [1,5,10]:
 ic=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q))
 x=np.asarray(ic); print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
 print('yearly', {int(y):round(x[np.array([p.index[j].year for j in range(len(p)-h) if len(pd.concat([f.iloc[j],(p.iloc[j+h]/p.iloc[j]-1).rename("y")],axis=1).dropna())>=8 and pd.concat([f.iloc[j],(p.iloc[j+h]/p.iloc[j]-1).rename("y")],axis=1).dropna().iloc[:,0].nunique()>1])==y].mean(),5) for y in sorted(p.index.year.unique()) if sum(1 for j in range(len(p)-h) if p.index[j].year==y and len(pd.concat([f.iloc[j],(p.iloc[j+h]/p.iloc[j]-1).rename('y')],axis=1).dropna())>=8)>0})
print('assets',len(U),'valid factor dates',int(f.notna().any(axis=1).sum()))
print('turnover_proxy',round((f.rank(pct=True,axis=1).diff().abs().mean(axis=1)>0.25).mean(),4))
