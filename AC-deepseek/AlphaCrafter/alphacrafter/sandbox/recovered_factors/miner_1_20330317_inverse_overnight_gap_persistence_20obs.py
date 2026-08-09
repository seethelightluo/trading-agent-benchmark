"""Single-factor research: inverse overnight-gap persistence (20 observations)."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; rows={}
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
    rows[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a: rows[a]['close'].astype(float) for a in assets},axis=1).sort_index()
f=pd.DataFrame(index=close.index,columns=assets,dtype=float)
for a in assets:
    d=rows[a].reindex(close.index)
    # A run of positive overnight gaps is treated as an exhaustion/reversal signal.
    gap=d['open'].astype(float).div(d['close'].astype(float).shift(1)).sub(1)
    f[a]=-gap.rolling(20,min_periods=15).mean()
def evaluate(h):
    fw=close.shift(-h).div(close).sub(1); vals=[]; counts=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt].rename('factor'),fw.loc[dt].rename('forward')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8:
            vals.append((dt,z['factor'].corr(z['forward'],method='spearman')));counts.append(len(z))
    x=pd.Series(dict(vals),dtype=float)
    return x, {'dates':len(x),'ic':x.mean(),'icir':x.mean()/x.std(ddof=1),'hit_ratio':(x>0).mean(),'mean_instruments':float(np.mean(counts)),'min_instruments':int(min(counts))}
print('FACTOR inverse_overnight_gap_persistence_20obs = -mean_20(open_t/close_(t-1)-1)')
print('validation_end',close.index.max().date(),'start',close.index.min().date(),'universe',len(assets))
for h in [1,5,10,20]:
    x,m=evaluate(h);print('HORIZON',h,m)
    if h==20:
        for n,mask in [('2026_2028',x.index<'2029-01-01'),('2029_2030',(x.index>='2029-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01')]:
            y=x[mask];print('REGIME',n,{'dates':len(y),'ic':y.mean(),'icir':y.mean()/y.std(ddof=1),'hit_ratio':(y>0).mean()})
r=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(r)):
    z=r.iloc[[i-1,i]].T.dropna()
    if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('DIAGNOSTICS',{'coverage_cells':int(f.notna().sum().sum()),'total_cells':int(f.size),'coverage':float(f.notna().mean().mean()),'mean_rank_turnover':float(np.mean(turns)),'median_cs_iqr':float(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median())})
print('Library novelty audit intentionally not asserted: all admitted signals must be reconstructed and compared before any admission.')
