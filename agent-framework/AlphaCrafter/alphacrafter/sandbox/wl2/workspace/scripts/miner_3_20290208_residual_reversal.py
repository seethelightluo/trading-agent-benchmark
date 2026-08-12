import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in A:
 x=get_stock_daily_data(s,days=2400)
 if x is not None:D[s]=x.sort_values('date').set_index('date').close.astype(float)
common=sorted(set.intersection(*[set(x.index) for x in D.values()])); P=pd.DataFrame({s:D[s].reindex(common) for s in A},index=common).ffill(); R=P.pct_change(); M=R.mean(axis=1)
def fac(i,s):
 r=R[s].iloc[:i+1]; m=M.iloc[:i+1]; a=r.iloc[-30:]; b=m.iloc[-30:]
 beta=a.cov(b)/(b.var()+1e-8)
 resid=a-beta*b
 return -resid.iloc[-5:].sum()/(r.iloc[-30:].std()+1e-6)
ics=[];ns=[];tr=[];dates=[];prev=None
for i in range(35,len(P)-1):
 f=pd.Series({s:fac(i,s) for s in A});y=R.iloc[i+1];g=f.index[f.notna()&y.notna()]
 if len(g)>=8:
  ics.append(f[g].corr(y[g]));ns.append(len(g));dates.append(P.index[i]);q=f.rank(pct=True);tr.append(np.abs(q-(prev if prev is not None else q)).mean());prev=q
z=np.array(ics);print('dates',len(z),'meanN',np.mean(ns),'assets',len(A),'coverage',len(z)/len(P),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1)*np.sqrt(252),'hit',np.mean(z>0),'turnover',np.mean(tr))
for n,a in [('early',z[:len(z)//2]),('late',z[len(z)//2:])]:print(n,'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1)*np.sqrt(252))
for h in [1,5,10]:
 q=[]
 for i in range(35,len(P)-h):
  f=pd.Series({s:fac(i,s) for s in A});y=P.iloc[i+h]/P.iloc[i]-1;g=f.index[f.notna()&y.notna()]
  if len(g)>=8:q.append(f[g].corr(y[g]))
 print('decay',h,'IC',np.mean(q),'dates',len(q))
pd.DataFrame({'date':dates,'ic':z}).to_csv('scripts/miner_3_20290208_residual_reversal_signal.csv',index=False)
print('period',P.index[0],P.index[-1])
