import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index()
r=p.pct_change(); m=r.mean(axis=1)
# residual momentum: 20d return less rolling beta to cross-asset mean times benchmark return
cov=r.rolling(60,min_periods=40).cov(m); var=m.rolling(60,min_periods=40).var()
beta=cov.div(var,axis=0)
res=r.sub(beta.mul(m,axis=0)).rolling(20,min_periods=15).sum()
# signal lag handled using same-day close and forward next periods
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in p.index:
  z=pd.concat([res.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(vals); print('h',h,'dates',len(s),'meanN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',np.mean(s>0))
print('coverage',res.notna().sum(axis=1).mean()/15,'valid>=8',(res.notna().sum(axis=1)>=8).sum(),'turn10',res.rank(pct=True).diff(10).abs().mean(axis=1).mean())
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 fr=p.shift(-10)/p-1; z=[]
 for dt in p.loc[lo:hi].index:
  q=pd.concat([res.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print(lo,hi,len(z),np.mean(z) if z else np.nan,np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
