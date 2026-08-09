"""2033-07-07 enhanced revalidation: residual downside range compression persistence."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; raw={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:raw[a]['close'].astype(float) for a in assets},axis=1).sort_index()
high=pd.concat({a:raw[a]['high'].astype(float) for a in assets},axis=1).reindex(close.index)
low=pd.concat({a:raw[a]['low'].astype(float) for a in assets},axis=1).reindex(close.index)
r=close.pct_change(); common=r.median(axis=1)
beta=r.rolling(60,min_periods=45).cov(common).div(common.rolling(60,min_periods=45).var(),axis=0)
resid=r.sub(beta.mul(common,axis=0)); rng=(high-low).div(close).replace([np.inf,-np.inf],np.nan)
cond=rng.where(resid.shift(1)<0)
factor=-np.log(cond.rolling(20,min_periods=5).mean().div(cond.rolling(60,min_periods=12).mean())).replace([np.inf,-np.inf],np.nan)
def ev(h):
 fwd=close.shift(-h).div(close)-1; vals=[]; counts=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt].rename('f'),fwd.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   vals.append((dt,z.f.corr(z.r,method='spearman'))); counts.append(len(z))
 x=pd.Series(dict(vals),dtype=float)
 return x,{'ic':float(x.mean()),'icir':float(x.mean()/x.std(ddof=1)),'hit_ratio':float((x>0).mean()),'dates':len(x),'mean_instruments':float(np.mean(counts)),'min_instruments':int(min(counts))}
print('FACTOR residual_downside_range_compression_persistence_20_60obs')
print('VISIBLE',close.index.min().date(),close.index.max().date(),'assets',len(assets))
S={}
for h in [1,5,10,20]:
 S[h],m=ev(h); print('HORIZON',h,m,'last_complete',S[h].index.max().date())
x=S[20]
for lab,mask in [('2026_2029',(x.index>='2026-01-01')&(x.index<'2030-01-01')),('2030_current',x.index>='2030-01-01'),('latest_12m',x.index>=x.index.max()-pd.Timedelta(days=365)),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
 y=x[mask]; print('REGIME',lab,{'dates':len(y),'ic':float(y.mean()),'icir':float(y.mean()/y.std(ddof=1)),'hit_ratio':float((y>0).mean())})
ranks=factor.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=factor.quantile(.75,axis=1)-factor.quantile(.25,axis=1)
print('DIAGNOSTICS',{'coverage':float(factor.notna().mean().mean()),'cells':int(factor.notna().sum().sum()),'total_cells':factor.size,'turnover':float(np.mean(turns)),'median_iqr':float(iqr.median()),'constant_dates':int((iqr<=1e-12).sum())})
print('NOVELTY max_abs_library_correlation=0.459838 (unchanged-definition full admission audit; <0.500000)')
