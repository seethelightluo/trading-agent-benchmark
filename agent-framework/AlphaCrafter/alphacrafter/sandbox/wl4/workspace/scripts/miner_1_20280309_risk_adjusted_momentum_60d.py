import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for s in U}
for h in [1,5,10,20]:
 rows=[]
 for s,d in F.items():
  r=d.close.pct_change(); f=r.rolling(60,min_periods=45).sum()/r.rolling(20,min_periods=15).std(); y=d.close.shift(-h)/d.close-1
  rows.append(pd.DataFrame({'date':d.index,'f':f.to_numpy(),'y':y.to_numpy()}))
 z=pd.concat(rows,ignore_index=True).dropna(); a=[];ns=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: a.append(spearmanr(g.f,g.y).statistic);ns.append(len(g))
 a=np.array(a);print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('universe',len(U),'period',z.date.min().date(),z.date.max().date())
