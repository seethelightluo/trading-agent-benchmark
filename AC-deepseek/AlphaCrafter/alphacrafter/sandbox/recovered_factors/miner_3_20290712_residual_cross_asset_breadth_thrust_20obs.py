"""Single candidate: residualized cross-asset breadth thrust, PIT through 2029-07-11."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-07-11')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A}); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
peer=r.mean(axis=1)
# Continuous, all-asset breadth thrust: own 20d risk-adjusted return relative to contemporaneous universe breadth.
raw=(r.rolling(20,min_periods=15).sum().sub(peer.rolling(20,min_periods=15).sum(),axis=0))/vol
# residualize from standard trend, reversal and risk so this tests breadth-relative leadership rather than duplicate momentum
trend=(p/p.shift(20)-1)/vol
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
risk=-vol

def resid(y,cs):
 o=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[q.loc[d].rename(str(i)) for i,q in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]];o.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
f=resid(raw,[trend,rev,risk])
def met(h):
 fw=p.shift(-h)/p-1; qs=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: qs.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
 x=pd.Series(dict(qs)); out={}
 for n,m in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('2029',x.index.year==2029),('latest120',np.arange(len(x))>=len(x)-120)]:
  y=x[m]; out[n]={'dates':len(y),'ic':y.mean(),'icir':y.mean()/y.std() if y.std() else np.nan}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'ic':x.mean(),'icir':x.mean()/x.std(),'hit':(x>0).mean(),'dates':len(x),'mean_instruments':np.mean(ns),'coverage':int(f.count().sum())/f.size,'turnover_10d':np.mean(turn),'regimes':out}
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'cells',int(f.count().sum()),'possible',f.size)
for h in [1,5,10,20]: print('METRIC',json.dumps(met(h),default=float))
# mandatory maximum library screen against signal reconstructions / admitted proxies
lib={'trend':trend,'reversal':rev,'risk':risk}
mx=0; who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); rho=z.f.corr(z.x,method='spearman'); print('LIB',n,rho,len(z))
 if abs(rho)>mx: mx=abs(rho);who=n
print('MAX',mx,who)
