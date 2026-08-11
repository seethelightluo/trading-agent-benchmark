import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index(); p.columns=U; p=p.ffill()
r=p.pct_change(); med=r.median(axis=1); w=20; res=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i in range(w,len(p)):
 h=r.iloc[i-w:i]; mm=med.iloc[i-w:i]; v=mm.var(); beta=h.apply(lambda z:z.cov(mm)/v if v>1e-12 else 0)
 res.iloc[i]=((p.iloc[i]/p.iloc[i-w]-1)-beta*(np.prod(1+mm)-1)).values
vol=r.rolling(20).std(); shock=-r/(vol+1e-8); f=res.rank(axis=1,pct=True)*.65+shock.rank(axis=1,pct=True)*.35
for H in [1,5,10]:
 ic=[];ns=[]
 for i in range(len(p)-H):
  q=pd.concat([f.iloc[i],(p.iloc[i+H]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q))
 x=np.array(ic); print('H',H,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
print('assets',len(U),'date_range',p.index.min().date(),p.index.max().date())
