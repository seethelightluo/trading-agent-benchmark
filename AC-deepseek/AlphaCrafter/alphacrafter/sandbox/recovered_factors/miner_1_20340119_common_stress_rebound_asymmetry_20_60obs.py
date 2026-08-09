"""Candidate: common-stress rebound asymmetry, 20 observations.
Measures an asset's normalized subsequent return after a lagged common-market
stress day minus its response after a lagged common-market strength day.  Both
states and responses are completed-bar information at the score date.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets=get_account_dict()['watch_list']; raw={}
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
    raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:raw[a]['close'].astype(float) for a in assets},axis=1).sort_index()
r=close.pct_change(); common=r.median(axis=1)
# State at t-1 determines asset response at t. Thresholds are rolling prior
# completed common returns, shifted one day so a state threshold never uses t.
qlo=common.rolling(60,min_periods=45).quantile(.25).shift(1)
qhi=common.rolling(60,min_periods=45).quantile(.75).shift(1)
stress=(common.shift(1)<=qlo).astype(float).where(qlo.notna())
strength=(common.shift(1)>=qhi).astype(float).where(qhi.notna())
# Conditional means use a 20-observation trailing completed response window.
# Minimum 3 events in each state prevents one-event scores; scale by own realized risk.
def condmean(mask):
    num=r.mul(mask,axis=0).rolling(20,min_periods=15).sum()
    den=mask.rolling(20,min_periods=15).sum()
    return num.div(den.where(den>=3),axis=0)
post_stress=condmean(stress); post_strength=condmean(strength)
vol=r.rolling(20,min_periods=15).std().clip(lower=1e-5)
f=(post_stress-post_strength).div(vol).replace([np.inf,-np.inf],np.nan)

def evaluate(h):
    fw=close.shift(-h).div(close).sub(1); out=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt].rename('factor'),fw.loc[dt].rename('forward')],axis=1).dropna()
        if len(z)>=8 and z.factor.nunique()>1 and z.forward.nunique()>1:
            out.append((dt,z.factor.corr(z.forward,method='spearman'))); ns.append(len(z))
    x=pd.Series(dict(out)); sd=x.std(ddof=1)
    return x,dict(dates=len(x),ic=x.mean(),icir=x.mean()/sd,hit_ratio=(x>0).mean(),mean_instruments=float(np.mean(ns)),min_instruments=min(ns))
print('FACTOR common_stress_rebound_asymmetry_20_60obs')
print('expression [mean_20(r_i,t | median_r,t-1 <= q25_60)-mean_20(r_i,t | median_r,t-1 >= q75_60)] / std_20(r_i)')
print('visible_history',close.index.min().date(),close.index.max().date(),'assets',len(assets))
allx={}
for h in (1,5,10,20):
 x,m=evaluate(h); allx[h]=x; print('HORIZON',h,m)
 if h==10:
  for name,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
   y=x[mask]; print('REGIME',name,'dates',len(y),'ic',y.mean(),'icir',y.mean()/y.std(ddof=1),'hit_ratio',(y>0).mean())
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('coverage',f.notna().mean().mean(),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'daily_rank_turnover',float(np.mean(turns)),'median_iqr',iqr.median(),'constant_dates',int((iqr<=1e-12).sum()))
print('Library signal Spearman audit deferred unless same-horizon IC/ICIR admission gates pass.')
