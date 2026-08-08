"""Candidate: slow residual cross-asset breadth persistence, PIT through 2029-07-25."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-07-25')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A}); r=p.pct_change(); peer=r.mean(axis=1)
# 120-day persistence of positive relative-return participation, smoothed by a 20-day mean.
b=(r.gt(peer,axis=0).astype(float).rolling(120,min_periods=80).mean()).rolling(20,min_periods=10).mean()
vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(60)-1)/vol; risk=-vol
# residualize cross-sectionally against trend and risk, isolating breadth persistence.
f=b*np.nan
for d in b.index:
 z=pd.concat([b.loc[d].rename('y'),trend.loc[d].rename('trend'),risk.loc[d].rename('risk')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1:]]; f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def calc(h):
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));
 out={'horizon':h,'ic':x.mean(),'icir':x.mean()/x.std(),'hit':(x>0).mean(),'dates':len(x),'mean_n':np.mean(ns)}
 for label,mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('2029',x.index.year==2029),('latest120',np.arange(len(x))>=len(x)-120)]:
  y=x[mask];out[label]={'dates':len(y),'ic':y.mean(),'icir':y.mean()/y.std() if len(y)>1 else np.nan}
 return out
print('VISIBLE',END.date(),'assets',len(A),'dates',len(p),'cells',int(f.count().sum()),'possible',f.size,'coverage',f.count().sum()/f.size)
for h in [1,5,10,20]: print('METRIC',json.dumps(calc(h),default=float))
# Explicit evidence for the signals used in residualization; full library evidence is required before admission.
mx=0;who=''
for n,x in {'risk':risk,'trend':trend}.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z));
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_PARTIAL',mx,who)
