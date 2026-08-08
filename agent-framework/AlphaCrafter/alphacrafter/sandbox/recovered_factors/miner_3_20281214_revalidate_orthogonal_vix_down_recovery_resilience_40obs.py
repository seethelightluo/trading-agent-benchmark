"""Revalidation only: Orthogonal VIX-down-day recovery resilience, 40 observations."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-12-13')
def read(a,index=False):
 d=(get_index_daily_data(a,5000) if index else get_stock_daily_data(a,5000)).set_index('date'); d.index=pd.to_datetime(d.index); return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:read(a) for a in A}); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def resid(y,controls):
 out=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[q.loc[d].rename(str(i)) for i,q in enumerate(controls)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]; out.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A}); kurt=-r.rolling(40,min_periods=30).kurt()
vix=read('VIX',True).pct_change().reindex(r.index)
upraw=pd.DataFrame({a:r[a].where(vix>0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
raw=pd.DataFrame({a:r[a].where(vix<0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
f=resid(raw,[es,down,kurt,trend,upraw])
def one(h):
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
 x=pd.Series(dict(vals)); sd=x.std()
 reg={}
 for nm,sel in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('latest_120',np.arange(len(x))>=len(x)-120)]:
  q=x[sel]; reg[nm]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if q.std() else np.nan}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'ic':x.mean(),'icir':x.mean()/sd,'hit':(x>0).mean(),'dates':len(x),'se':sd/np.sqrt(len(x)),'mean_n':np.mean(ns),'turnover':np.mean(turn),'regimes':reg}
print('REVALIDATION visible_through=',END.date(),'assets=',len(A),'sessions=',len(p),'cells=',int(f.count().sum()),'/',f.size)
for h in [1,5,10,20]: print('H',h,json.dumps(one(h),default=float))
# Recorded admission orthogonality is rechecked against the closest admitted factor: acceleration.
acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol; orth=resid(acc,[trend]); z=pd.concat([f.stack(),orth.stack()],axis=1).dropna(); print('CORR closest_admitted_acceleration',z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),'cells',len(z))
PY
