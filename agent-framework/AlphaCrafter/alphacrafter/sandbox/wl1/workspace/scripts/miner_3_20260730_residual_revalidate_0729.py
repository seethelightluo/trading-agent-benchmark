import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-29')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index()
r=p.pct_change(); m=r.median(axis=1)
# residual 20-session momentum: trailing return purged of cross-sectional median return scaled by trailing beta
w=20; f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i in range(w,len(p)):
 h=r.iloc[i-w:i]; mm=m.iloc[i-w:i]; var=mm.var()
 beta=h.apply(lambda z:z.cov(mm)/var if var>1e-12 else 0)
 f.iloc[i]=(p.iloc[i]/p.iloc[i-w]-1)-beta*(np.prod(1+mm)-1)
for horizon in [1,5,10]:
 ic=[]; ns=[]; rr=[]
 for i in range(len(p)-horizon):
  q=pd.concat([f.iloc[i],(p.iloc[i+horizon]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q))
 x=np.array(ic); print('horizon',horizon,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
vals=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i],(p.iloc[i+1]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(q)>=8: vals.append((f.index[i],spearmanr(q.iloc[:,0],q.y).statistic))
z=pd.Series(dict(vals)); print('regime', {int(y):round(z[z.index.year==y].mean(),5) for y in sorted(z.index.year.unique())})
print('assets',len(U),'valid_dates',len(z))
