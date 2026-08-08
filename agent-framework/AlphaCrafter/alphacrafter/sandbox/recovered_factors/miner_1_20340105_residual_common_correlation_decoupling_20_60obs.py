"""Candidate: residual common-correlation decoupling (20 versus 60 days).
High values identify assets whose absolute dependence on the daily cross-asset median
return recently fell relative to their own 60-session baseline.  It is a simple,
lagged diversification/state-transition measure rather than a directional return rule.
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
# Correlations and all factor observations are shifted, so the signal only uses
# information through the preceding completed session when paired to a return.
c20=r.rolling(20,min_periods=15).corr(common).unstack().reindex(columns=assets)
c60=r.rolling(60,min_periods=45).corr(common).unstack().reindex(columns=assets)
f=(c60.abs()-c20.abs()).div(c60.abs().clip(lower=.05)).shift(1).replace([np.inf,-np.inf],np.nan)
def evaluate(h):
    forward=close.shift(-h).div(close).sub(1); obs=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt].rename('factor'),forward.loc[dt].rename('forward')],axis=1).dropna()
        if len(z)>=8 and z.factor.nunique()>1 and z.forward.nunique()>1:
            obs.append((dt,z.factor.corr(z.forward,method='spearman'))); ns.append(len(z))
    x=pd.Series(dict(obs)); sd=x.std(ddof=1)
    return x,{'dates':len(x),'ic':x.mean(),'icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'mean_instruments':float(np.mean(ns)),'min_instruments':min(ns)}
print('FACTOR residual_common_correlation_decoupling_20_60obs')
print('expression lag1((abs(corr_60(asset_return,median_return))-abs(corr_20(asset_return,median_return)))/max(abs(corr_60),0.05))')
print('visible_history',close.index.min().date(),close.index.max().date(),'assets',len(assets))
for h in (1,5,10,20):
 x,m=evaluate(h); print('HORIZON',h,m)
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
