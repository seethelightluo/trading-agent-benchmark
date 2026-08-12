import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in A:
 x=get_stock_daily_data(s,days=2400)
 if x is not None:D[s]=x.sort_values('date').set_index('date').close.astype(float)
common=sorted(set.intersection(*[set(D[s].index) for s in A]));P=pd.DataFrame({s:D[s].reindex(common) for s in A},index=common).ffill();R=P.pct_change();M=R.mean(axis=1)
def fct(i,s,L,T,V,k):
 a=R[s].iloc[i-L+1:i+1];b=M.iloc[i-L+1:i+1]; beta=a.cov(b)/(b.var()+1e-10);e=a-beta*b
 breadth=(R.iloc[i-9:i+1]>0).mean(axis=1).mean(); atten=1-k*abs(2*breadth-1)
 return e.iloc[-T:].sum()/(e.iloc[-V:].std()+1e-6)*atten
for L,T,V,k in [(60,15,20,.5),(60,25,20,.5),(60,20,30,.5),(60,20,20,.3),(40,20,20,.3),(90,20,20,.3)]:
 z=[];ns=[]
 for i in range(L+2,len(P)-1):
  x=pd.Series({s:fct(i,s,L,T,V,k) for s in A});y=R.iloc[i+1];g=x.index[x.notna()&y.notna()]
  if len(g)>=8:z.append(x[g].corr(y[g]));ns.append(len(g))
 z=np.array(z);print('params',L,T,V,k,'dates',len(z),'avgN',np.mean(ns),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1)*np.sqrt(252),'hit',np.mean(z>0))
print('period',P.index[0],P.index[-1])
