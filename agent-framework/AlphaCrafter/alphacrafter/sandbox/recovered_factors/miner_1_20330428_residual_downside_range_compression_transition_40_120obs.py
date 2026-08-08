"""Candidate: residual downside range-compression transition, longer 40/120 state windows.
High scores identify assets whose intraday range after idiosyncratic down days has
contracted materially versus a slow baseline, normalized by its baseline variability.
All conditioning information is lagged one completed day.
"""
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
r=close.pct_change(); median=r.median(axis=1)
# rolling beta to common cross-asset move; event label is shifted so range on the
# day after a residual-downside event is fully known before it enters the signal.
beta=r.rolling(60,min_periods=45).cov(median).div(median.rolling(60,min_periods=45).var(),axis=0)
resid=r.sub(beta.mul(median,axis=0))
rng=(high-low).div(close).replace([np.inf,-np.inf],np.nan)
event_rng=rng.where(resid.shift(1)<0)
short=event_rng.rolling(40,min_periods=10).mean(); long=event_rng.rolling(120,min_periods=30).mean()
# standardized proportional range compression, high = recent conditional calm.
f=(long-short).div(event_rng.rolling(120,min_periods=30).std()).replace([np.inf,-np.inf],np.nan)
def evaluate(h):
    forward=close.shift(-h).div(close).sub(1); obs=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt].rename('factor'),forward.loc[dt].rename('forward')],axis=1).dropna()
        if len(z)>=8 and z.factor.nunique()>1 and z.forward.nunique()>1:
            obs.append((dt,z.factor.corr(z.forward,method='spearman')));ns.append(len(z))
    x=pd.Series(dict(obs)); sd=x.std(ddof=1)
    return x,dict(dates=len(x),ic=x.mean(),icir=x.mean()/sd,hit_ratio=(x>0).mean(),mean_instruments=float(np.mean(ns)),min_instruments=min(ns))
print('FACTOR residual_downside_range_compression_transition_40_120obs')
print('expression (mean_120(range | resid[t-1]<0)-mean_40(range | resid[t-1]<0))/std_120(range | resid[t-1]<0)')
print('history',close.index.min().date(),close.index.max().date(),'assets',len(assets))
for h in (1,5,10,20):
    x,m=evaluate(h); print('HORIZON',h,m)
    if h==20:
        for name,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01')]:
            y=x[mask]; print('REGIME',name,'dates',len(y),'ic',y.mean(),'icir',y.mean()/y.std(ddof=1),'hit_ratio',(y>0).mean())
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
    z=ranks.iloc[[i-1,i]].T.dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('coverage',f.notna().mean().mean(),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'turnover',np.mean(turns),'median_iqr',iqr.median(),'constant_dates',int((iqr<=1e-12).sum()))
print('Library signal Spearman audit deferred unless same-horizon IC/ICIR admission gates pass.')
