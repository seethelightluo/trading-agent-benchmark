import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index(); r=p.pct_change(); med=r.median(axis=1)
# Cross-asset residual 20d momentum divided by idiosyncratic volatility; rewards persistent residual trends
w=20; f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i in range(w,len(p)):
 h=r.iloc[i-w:i]; mm=med.iloc[i-w:i]; var=mm.var(); beta=h.apply(lambda z:z.cov(mm)/var if var>1e-12 else 0)
 resid=h.subtract(mm,axis=0)-beta*0
 # residual cumulative return, beta-neutralized against median over window
 f.iloc[i]=((p.iloc[i]/p.iloc[i-w]-1)-beta*(np.prod(1+mm)-1))/(resid.std().replace(0,np.nan))
for horizon in [1,5,10]:
 ic=[]; ns=[]
 for i in range(len(p)-horizon):
  q=pd.concat([f.iloc[i],(p.iloc[i+horizon]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: ic.append(spearmanr(q.iloc[:,0],q.y).statistic); ns.append(len(q))
 x=np.array(ic); print('horizon',horizon,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
# annual 10d
ic=[]
for i in range(len(p)-10):
 q=pd.concat([f.iloc[i],(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(q)>=8: ic.append((f.index[i],spearmanr(q.iloc[:,0],q.y).statistic))
z=pd.Series(dict(ic)); print('annual10d',{int(y):round(z[z.index.year==y].mean(),5) for y in sorted(z.index.year.unique())})
print('assets',len(U),'valid_dates',len(z))
