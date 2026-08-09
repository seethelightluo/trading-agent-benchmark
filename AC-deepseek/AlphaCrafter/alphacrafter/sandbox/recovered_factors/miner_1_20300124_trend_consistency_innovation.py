import pandas as pd,numpy as np,json,glob
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2030-01-23')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A});r=p.pct_change();v=r.rolling(20,min_periods=14).std(); base=p.pct_change(20)/v
cons=((r>0).rolling(20,min_periods=14).mean()-.5); raw=(base*cons).shift(1); trend=base.shift(1)
f=pd.DataFrame(index=raw.index,columns=A,dtype=float)
for d in f.index:
 z=pd.concat([raw.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1].values]; b=np.linalg.lstsq(X,z.iloc[:,0].values,rcond=None)[0]
  f.loc[d,z.index]=z.iloc[:,0]-X@b
def calc(h):
 fw=p.shift(-h)/p-1;q=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(q)); turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 regs={}
 for n,mask in [('2020-24',x.index.year<=2024),('2025-27',(x.index.year>=2025)&(x.index.year<=2027)),('2028-29',(x.index.year>=2028)&(x.index.year<=2029)),('latest120',np.arange(len(x))>=len(x)-120)]:
  y=x[mask];regs[n]=[len(y),float(y.mean()) if len(y) else None,float(y.mean()/y.std()) if len(y)>1 else None]
 return {'horizon':h,'dates':len(x),'mean_n':float(np.mean(ns)),'ic':float(x.mean()),'icir':float(x.mean()/x.std()),'hit':float((x>0).mean()),'turnover10':float(np.mean(turn)),'regimes':regs}
print('assets',len(A),'dates',len(p),'signal_coverage',float(f.count().sum()/f.size))
for h in [1,5,10,20]: print(json.dumps(calc(h)))
