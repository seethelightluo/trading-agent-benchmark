"""2033-05-26 scheduled revalidation of one admitted residual-downside range factor."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets = get_account_dict()['watch_list']
raw = {}
for a in assets:
    d = get_stock_daily_data(a, 5000).copy()
    d['date'] = pd.to_datetime(d['date'])
    raw[a] = d.drop_duplicates('date').set_index('date').sort_index()
close = pd.concat({a: raw[a]['close'].astype(float) for a in assets}, axis=1).sort_index()
high = pd.concat({a: raw[a]['high'].astype(float) for a in assets}, axis=1).reindex(close.index)
low = pd.concat({a: raw[a]['low'].astype(float) for a in assets}, axis=1).reindex(close.index)
r = close.pct_change(); median_return = r.median(axis=1)
beta = r.rolling(60, min_periods=45).cov(median_return).div(median_return.rolling(60, min_periods=45).var(), axis=0)
residual_return = r.sub(beta.mul(median_return, axis=0))
intraday_range = (high-low).div(close).replace([np.inf, -np.inf], np.nan)
conditional_range = intraday_range.where(residual_return.shift(1) < 0)
factor = -np.log(conditional_range.rolling(20, min_periods=5).mean().div(conditional_range.rolling(60, min_periods=12).mean())).replace([np.inf,-np.inf],np.nan)

def evaluate(horizon):
    forward = close.shift(-horizon).div(close)-1
    rows=[]; ns=[]
    for dt in factor.index:
        pair=pd.concat([factor.loc[dt].rename('factor'),forward.loc[dt].rename('forward')],axis=1).dropna()
        if len(pair)>=8 and pair.factor.nunique()>1 and pair.forward.nunique()>1:
            rows.append((dt,pair.factor.corr(pair.forward,method='spearman'))); ns.append(len(pair))
    x=pd.Series(dict(rows),dtype=float)
    return x, dict(ic=x.mean(),icir=x.mean()/x.std(ddof=1),hit_ratio=(x>0).mean(),dates=len(x),mean_instruments=float(np.mean(ns)),min_instruments=int(min(ns)))

print('FACTOR residual_downside_range_compression_persistence_20_60obs')
print('VISIBLE',close.index.min().date(),close.index.max().date(),'assets',len(assets))
series={}
for h in (1,5,10,20):
    series[h], metrics=evaluate(h); print('HORIZON',h,metrics)
x=series[20]
for label,mask in [('2026_2029',x.index<'2030-01-01'),('2030_current',x.index>='2030-01-01'),('latest_12m',x.index>=x.index.max()-pd.Timedelta(days=365))]:
    y=x[mask]; print('REGIME',label,'dates',len(y),'ic',y.mean(),'icir',y.mean()/y.std(ddof=1),'hit_ratio',(y>0).mean())
ranks=factor.rank(axis=1,pct=True); turnover=[]
for i in range(1,len(ranks)):
    z=ranks.iloc[[i-1,i]].T.dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turnover.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=factor.quantile(.75,axis=1)-factor.quantile(.25,axis=1)
print('DIAGNOSTICS coverage',factor.notna().mean().mean(),'cells',int(factor.notna().sum().sum()),'of',factor.size,'turnover',np.mean(turnover),'median_iqr',iqr.median(),'constant_dates',int((iqr<=1e-12).sum()))
print('NOVELTY unchanged factor: previous complete-library audit max_abs_rho=0.459838 (<0.5); no new factor admission is requested.')
