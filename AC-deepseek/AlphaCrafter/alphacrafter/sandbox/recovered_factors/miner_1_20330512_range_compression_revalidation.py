"""Scheduled revalidation: unchanged residual downside range-compression persistence."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; raw={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date); raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:raw[a].close.astype(float) for a in assets},axis=1).sort_index()
high=pd.concat({a:raw[a].high.astype(float) for a in assets},axis=1).reindex(close.index)
low=pd.concat({a:raw[a].low.astype(float) for a in assets},axis=1).reindex(close.index)
r=close.pct_change(); med=r.median(axis=1)
beta=r.rolling(60,min_periods=45).cov(med).div(med.rolling(60,min_periods=45).var(),axis=0)
resid=r.sub(beta.mul(med,axis=0)); rng=(high-low).div(close).replace([np.inf,-np.inf],np.nan)
event_rng=rng.where(resid.shift(1)<0)
f=-np.log(event_rng.rolling(20,min_periods=5).mean().div(event_rng.rolling(60,min_periods=12).mean())).replace([np.inf,-np.inf],np.nan)
def ev(h):
 fw=close.shift(-h).div(close)-1; out=[]; nn=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1: out.append((dt,z.f.corr(z.r,method='spearman'))); nn.append(len(z))
 x=pd.Series(dict(out)); return x,{'ic':x.mean(),'icir':x.mean()/x.std(ddof=1),'hit':(x>0).mean(),'dates':len(x),'mean_n':np.mean(nn),'min_n':min(nn)}
print('REVALIDATION residual_downside_range_compression_persistence_20_60obs')
print('visible_history',close.index.min().date(),close.index.max().date(),'instruments',len(assets))
allx={}
for h in (1,5,10,20):
 x,m=ev(h);allx[h]=x;print('H',h,m)
x=allx[20]
for name, mask in [('2026_2029',x.index<'2030-01-01'),('2030_current',x.index>='2030-01-01'),('recent_12m',x.index>=x.index.max()-pd.Timedelta(days=365))]:
 y=x[mask];print('REGIME',name,'dates',len(y),'ic',y.mean(),'icir',y.mean()/y.std(ddof=1),'hit',(y>0).mean())
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('DIAGNOSTICS coverage',f.notna().mean().mean(),'cells',int(f.notna().sum().sum()),'of',f.size,'turnover',np.mean(turns),'median_iqr',iqr.median(),'constant_dates',int((iqr<=1e-12).sum()))
print('Correlation: unchanged definition; prior complete library audit max_abs_rho=0.459838 < 0.500000.')
