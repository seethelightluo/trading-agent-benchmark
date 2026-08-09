"""Single-idea research: directional intraday-efficiency persistence (20 observations).
Score is the rolling mean signed close-to-open move scaled by each bar's high-low range.
It distinguishes persistent intraday demand from raw multi-day price momentum."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets=get_account_dict()['watch_list']; rows={}
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
    rows[a]=d.drop_duplicates('date').set_index('date').sort_index()
# aligned close and factor; avoid using any field beyond API's visible history
close=pd.concat({a:rows[a]['close'].astype(float) for a in assets},axis=1).sort_index()
f=pd.DataFrame(index=close.index,columns=assets,dtype=float)
for a in assets:
    d=rows[a].reindex(close.index)
    rng=(d['high'].astype(float)-d['low'].astype(float)).replace(0,np.nan)
    signed=(d['close'].astype(float)-d['open'].astype(float))/rng
    f[a]=signed.rolling(20,min_periods=15).mean()

def evaluate(h):
    forward=close.shift(-h)/close-1; vals=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt].rename('factor'),forward.loc[dt].rename('forward')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8: vals.append((dt,z.factor.corr(z.forward,method='spearman'))); ns.append(len(z))
    x=pd.Series(dict(vals)); sd=x.std(ddof=1)
    return x, {'dates':len(x),'ic':x.mean(),'icir':x.mean()/sd,'hit':(x>0).mean(),'mean_n':np.mean(ns),'min_n':min(ns)}
print('FACTOR directional_intraday_efficiency_persistence_20obs = mean_20((close-open)/(high-low))')
print('history',close.index.min().date(),close.index.max().date(),'universe',len(assets))
for h in (1,5,10,20):
    x,m=evaluate(h); print('H',h, m)
    if h==20:
        for name,mask in [('2026_2028',x.index<'2029-01-01'),('2029_2030',(x.index>='2029-01-01')&(x.index<'2031-01-01')),('2031_2033',x.index>='2031-01-01')]:
            y=x[mask]; print('REGIME',name,'dates',len(y),'IC',y.mean(),'ICIR',y.mean()/y.std(ddof=1),'hit',(y>0).mean())
rank=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rank)):
    z=rank.iloc[[i-1,i]].T.dropna()
    if len(z)>=8: ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('coverage',int(f.notna().sum().sum()),'/',f.size,'=',f.notna().mean().mean(),'turnover',np.mean(ts),'median_iqr',f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median())
print('NOTE: no persistence attempted in this script; a passing signal requires a separate exact 30-factor library signal audit.')
