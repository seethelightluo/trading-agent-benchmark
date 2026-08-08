"""Candidate: smoothed correlation-conditioned residual trend (40/90 observations)."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']; raw={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:raw[a]['close'].astype(float) for a in assets},axis=1,sort=True).sort_index()
r=close.pct_change(); market=r.median(axis=1)
beta=r.rolling(90,min_periods=65).cov(market).div(market.rolling(90,min_periods=65).var(),axis=0)
residual=r.sub(beta.mul(market,axis=0)); trend=residual.rolling(40,min_periods=30).sum()
def paircorr(x):
 c=x.corr(min_periods=10).to_numpy(float); np.fill_diagonal(c,np.nan); return np.nanmean(c)
state=pd.Series(np.nan,index=r.index)
for i in range(29,len(r)): state.iloc[i]=paircorr(r.iloc[i-29:i+1])
baseline=state.rolling(90,min_periods=65).median().shift(1)
scale=state.rolling(90,min_periods=65).std().shift(1).clip(lower=1e-4)
# Smooth continuous low-correlation state rather than a clipped binary-like state
condition=np.tanh((baseline-state)/scale)
f=trend.mul(condition,axis=0).replace([np.inf,-np.inf],np.nan)
def ev(h):
 fw=close.shift(-h).div(close)-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1: vals.append((dt,z.x.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals)); return x,dict(dates=len(x),ic=x.mean(),icir=x.mean()/x.std(ddof=1),hit_ratio=(x>0).mean(),mean_instruments=float(np.mean(ns)),min_instruments=min(ns))
print('FACTOR smoothed_correlation_conditioned_residual_trend_40_90obs')
print('expression sum_40(r_i-beta_90,i*median_return)*tanh((median_90(avg_paircorr_30)-avg_paircorr_30)/std_90(avg_paircorr_30))')
print('visible_history',close.index.min().date(),close.index.max().date(),'assets',len(assets))
for h in [1,5,10,20]:
 x,s=ev(h);print('HORIZON',h,s)
 if h==10:
  for n,q in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
   y=x[q];print('REGIME',n,'dates',len(y),'ic',y.mean(),'icir',y.mean()/y.std(ddof=1),'hit_ratio',(y>0).mean())
ranks=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('coverage',f.notna().mean().mean(),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'daily_rank_turnover',np.mean(turn),'median_iqr',iqr.median(),'constant_dates',int((iqr<=1e-12).sum()))
