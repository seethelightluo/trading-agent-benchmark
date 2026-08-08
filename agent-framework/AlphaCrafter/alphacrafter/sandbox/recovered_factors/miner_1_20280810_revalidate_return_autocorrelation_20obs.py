"""One idea: timely revalidation of 20-observation return autocorrelation.
Higher serial dependence is hypothesized to predict intermediate-horizon continuation."""
import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-08-09')
def load(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:load(a) for a in A}); r=p.pct_change()
# corr(r_t,r_t-1) among the latest 20 returns, needing 16 paired observations
f=r.rolling(20,min_periods=16).corr(r.shift(1))
def metrics(h):
 fw=p.shift(-h)/p-1; vals=[]; sizes=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));sizes.append(len(z))
 x=pd.Series(dict(vals),dtype=float); x.index=pd.to_datetime(x.index)
 def stat(q):
  s=q.std();return {'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/s) if len(q)>1 and s>0 else None,'hit_ratio':float((q>0).mean()) if len(q) else None}
 regimes={str(y):stat(x[x.index.year==y]) for y in range(2020,2029)}
 regimes['latest_120']=stat(x.tail(120))
 t=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:t.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 s=x.std()
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/s),'ic_se':float(s/np.sqrt(len(x))),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'mean_instruments':float(np.mean(sizes)),'coverage_cells':int(f.count().sum()),'coverage_fraction':float(f.count().sum()/f.size),'turnover_10d':float(np.mean(t)),'regimes':regimes}
print('FACTOR return_autocorrelation_20obs; visible through',END.date(),'assets',len(A),'data',p.index.min().date(),p.index.max().date())
for h in [1,5,10,20]:print('METRIC',json.dumps(metrics(h)))
# Exact active volume formula is tested as mandatory evidence; it has known constant / invalid support.
v=pd.DataFrame({a:pd.to_numeric(get_stock_daily_data(a,5000).set_index('date').loc[:END,'volume'],errors='coerce') for a in A})
rv=np.log(v/v.rolling(20,min_periods=15).mean())
z=pd.concat([f.stack().rename('candidate'),rv.stack().rename('relative_volume')],axis=1).dropna()
print('VOLUME_LIBRARY_CORRELATION',f'{z.candidate.corr(z.relative_volume,method="spearman")}', 'overlap_cells',len(z),'volume_unique',z.relative_volume.nunique())
