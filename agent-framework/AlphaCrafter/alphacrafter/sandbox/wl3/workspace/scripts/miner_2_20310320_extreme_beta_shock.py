import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff();b=r.mean(axis=1);beta=r.rolling(60).cov(b).div(b.rolling(60).var()+1e-12,axis=0);e=r-beta.mul(b,axis=0);z=e.rolling(5).sum().div(e.rolling(40).std()*np.sqrt(5)+1e-12)
# Only fade shocks that are extreme relative to each instrument's own residual history; otherwise remain neutral.
f=(-z.where(z.abs()>=1.25)).shift(1)
R=[]
for i,t in enumerate(p.index[:-1]):
 q=pd.concat([f.loc[t],r.iloc[i+1]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:R.append((t,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
x=pd.DataFrame(R,columns=['date','n','ic']);print('dates',len(x),'avgN',round(x.n.mean(),2),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),4),'coverage',round(f.notna().stack().mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2032')]:
 q=x[(x.date.astype(str)>=a)&(x.date.astype(str)<=b)];print(a+'-'+b,'dates',len(q),'IC',round(q.ic.mean(),6) if len(q) else None)
