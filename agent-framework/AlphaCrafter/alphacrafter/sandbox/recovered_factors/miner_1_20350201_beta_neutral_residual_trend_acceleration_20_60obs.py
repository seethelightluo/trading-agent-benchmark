"""Candidate: beta-neutral residual trend acceleration (20d versus 60d), tested through 2035-01-31."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']; raw={}
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
    raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:x.close.astype(float) for a,x in raw.items()},axis=1).sort_index()
r=close.pct_change(); mkt=r.median(axis=1)
beta=r.rolling(60,min_periods=45).cov(mkt).div(mkt.rolling(60,min_periods=45).var(),axis=0)
res=r.sub(beta.mul(mkt,axis=0))
# Positive values identify assets whose beta-neutral 20d trend exceeds their slower 60d baseline.
f=res.rolling(20,min_periods=15).sum()-res.rolling(60,min_periods=45).sum()/3
f=f.div(res.rolling(60,min_periods=45).std().replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
def evaluate(h):
 fw=close.shift(-h).div(close).sub(1); out=[]; n=[]
 for dt in f.index:
  z=pd.concat((f.loc[dt].rename('factor'),fw.loc[dt].rename('forward')),axis=1).dropna()
  if len(z)>=8 and z.factor.nunique()>1 and z.forward.nunique()>1:
   out.append((dt,z.factor.corr(z.forward,method='spearman'))); n.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1)
 return x,{'ic':x.mean(),'icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_instruments':float(np.mean(n)),'min_instruments':min(n)}
print('FACTOR beta_neutral_residual_trend_acceleration_20_60obs')
print('EXPRESSION (sum_20(r_i-beta_60,i*r_median)-sum_60(r_i-beta_60,i*r_median)/3)/std_60(residual_return_i)')
print('VISIBLE',close.index.min().date(),close.index.max().date(),'assets',len(assets))
for h in (1,5,10,20):
 x,s=evaluate(h); print('HORIZON',h,s)
 if h==10:
  for label,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
   q=x[mask]; print('REGIME',label,'dates',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1),'hit_ratio',(q>0).mean())
rk=f.rank(axis=1,pct=True); turns=[]
for j in range(1,len(rk)):
 z=rk.iloc[[j-1,j]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('COVERAGE',f.notna().mean().mean(),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'turnover',float(np.mean(turns)),'median_iqr',iqr.median(),'constant_dates',int((iqr<=1e-12).sum()))
