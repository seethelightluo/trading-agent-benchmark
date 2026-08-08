"""Single candidate: volume-shock short reversal, PIT through 2030-02-06."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2030-02-06')
def get(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
D={a:get(a) for a in A}; p=pd.DataFrame({a:pd.to_numeric(D[a].close,errors='coerce') for a in A}); r=p.pct_change();
vol=pd.DataFrame({a:pd.to_numeric(D[a].volume,errors='coerce') for a in A}).replace(0,np.nan)
# A contrarian signal: recent loss, scaled by volatility, amplified only by unusual but not
# isolated trading activity. All windows use completed observations.
rv=r.rolling(20,min_periods=15).std(); zvol=np.log(vol).sub(np.log(vol).rolling(40,min_periods=25).mean())/np.log(vol).rolling(40,min_periods=25).std()
raw=-(p/p.shift(5)-1)/rv * np.tanh(zvol.clip(-3,3)/2)
# residualize only cross-sectionally against plain short reversal to isolate volume information
base=-(p/p.shift(5)-1)/rv
f=raw* np.nan
for d in raw.index:
 z=pd.concat([raw.loc[d].rename('y'),base.loc[d].rename('b')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.b]; f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def met(h):
 fw=p.shift(-h)/p-1; q=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(q)); turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'dates':len(x),'ic':x.mean(),'icir':x.mean()/x.std(),'hit':(x>0).mean(),'mean_instruments':np.mean(ns),'coverage':f.count().sum()/f.size,'turnover_10d':np.mean(turn),'regimes':{str(y):{'dates':int((x.index.year==y).sum()),'ic':x[x.index.year==y].mean(),'icir':x[x.index.year==y].mean()/x[x.index.year==y].std()} for y in range(2026,2031)}}
print('VISIBLE',END.date(),'assets',len(A),'dates',len(p),'cells',int(f.count().sum()),'possible',f.size)
for h in [1,5,10,20]:print('METRIC',json.dumps(met(h),default=float))
# broad independent library screen; exact admitted signals are unavailable as persisted values,
# so this is conservative evidence against their principal constructions.
libs={'short_reversal':base,'momentum20':p.pct_change(20)/rv,'volatility':-rv,'volume_z':zvol,'trend60':p.pct_change(60)/rv}
mx=0;who=''
for n,x in libs.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,float(rho),len(z))
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',float(mx),who)
