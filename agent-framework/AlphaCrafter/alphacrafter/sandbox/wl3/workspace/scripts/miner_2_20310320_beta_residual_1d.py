import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff();b=r.mean(axis=1);beta=r.rolling(60).cov(b).div(b.rolling(60).var()+1e-12,axis=0);e=r-beta.mul(b,axis=0);v=e.rolling(40).std()+1e-12
# One-day beta-residual reversal, lagged one completed session.
f=(-e/v).shift(1); rows=[]
for h in [1,3,5]:
 for i,t in enumerate(p.index[:-h]):
  q=pd.concat([f.loc[t],r.iloc[i+1:i+h+1].sum()],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:rows.append((t,h,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
x=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in [1,3,5]:
 q=x[x.h==h];print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
print('coverage',round(f.notna().stack().mean(),4),'assets',len(P))
