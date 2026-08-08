"""One interpretable idea: residual downside low-close repair efficiency.
After an asset suffers idiosyncratic downside and closes low in its range, measure its
next-session residual recovery; aggregate prior fully observed events over 20 sessions.
Higher values indicate repeatedly repaired downside shocks rather than persistent damage.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']; raw={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date)
 raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:raw[a].close.astype(float) for a in assets},axis=1).sort_index()
hi=pd.concat({a:raw[a].high.astype(float) for a in assets},axis=1).reindex(close.index)
lo=pd.concat({a:raw[a].low.astype(float) for a in assets},axis=1).reindex(close.index)
r=close.pct_change(); med=r.median(axis=1); resid=r.sub(med,axis=0)
loc=(close-lo).div((hi-lo).replace(0,np.nan)).clip(0,1)
# Event severity is known on event day. Its one-day residual repair is shifted so no
# future observation is incorporated in the date-t signal.
severity=(-resid.clip(upper=0))*(1-loc)
repair=resid.shift(-1)
observed=(severity*repair).shift(1)
scale=severity.shift(1).rolling(20,min_periods=15).sum().replace(0,np.nan)
f=observed.rolling(20,min_periods=15).sum().div(scale).div(r.rolling(20,min_periods=15).std().replace(0,np.nan))
def ev(h):
 fw=close.shift(-h).div(close).sub(1); vals=[]; sizes=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append((dt,z.f.corr(z.y,method='spearman')));sizes.append(len(z))
 x=pd.Series(dict(vals)); sd=x.std(ddof=1)
 return x,{'dates':len(x),'ic':x.mean(),'icir':x.mean()/sd,'hit':(x>0).mean(),'mean_instruments':float(np.mean(sizes)),'min_instruments':min(sizes)}
print('FACTOR residual_downside_lowclose_repair_efficiency_20obs')
print('expression sum20_lag1(residual_return_next_day*max(median_return-return,0)*(1-close_location))/sum20_lag1(max(median_return-return,0)*(1-close_location))/std20(return)')
print('history',close.index.min().date(),close.index.max().date(),'assets',len(assets))
for h in (1,5,10,20):
 x,m=ev(h);print('H',h,m)
 if h==20:
  for n,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01')]:
   y=x[mask];print('REGIME',n,'dates',len(y),'ic',y.mean(),'icir',y.mean()/y.std(ddof=1),'hit',(y>0).mean())
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('coverage',f.notna().mean().mean(),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'turnover',np.mean(turn),'median_iqr',iqr.median(),'constant_dates',int((iqr<=1e-12).sum()))
print('Novelty correlation audit required only if IC and ICIR gates pass.')
