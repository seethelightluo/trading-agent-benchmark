import os, json, glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

# One idea: VIX-shock conditional resilience.  On each day score the asset's
# 60-observation mean return on high-VIX-change days minus its normal-VIX mean;
# higher is resilience. Remove ordinary trend/volatility exposures cross-sectionally.
END=5000
acc=get_account_dict(); A=list(acc['watch_list'])
def load_asset(a):
    x=get_stock_daily_data(a, END).copy(); x['date']=pd.to_datetime(x['date']); return x.set_index('date').sort_index()
raw={a:load_asset(a) for a in A}
close=pd.DataFrame({a:x['close'] for a,x in raw.items()}).sort_index()
ret=close.pct_change()
v=get_index_daily_data('VIX', END).copy(); v['date']=pd.to_datetime(v['date']); v=v.set_index('date').sort_index()
vcol='close' if 'close' in v else v.select_dtypes('number').columns[0]
dv=v[vcol].pct_change().reindex(close.index).ffill()
# threshold computed only from last 60 completed observations
shock=(dv > dv.rolling(60,min_periods=45).quantile(.80)).astype(float)
normal=(dv <= dv.rolling(60,min_periods=45).quantile(.80)).astype(float)
# conditional means; require at least 7 shock observations to prevent sparse unstable scores
shockmean=ret.mul(shock,axis=0).rolling(60,min_periods=45).sum().div(shock.rolling(60,min_periods=45).sum(),axis=0)
normmean=ret.mul(normal,axis=0).rolling(60,min_periods=45).sum().div(normal.rolling(60,min_periods=45).sum(),axis=0)
base=shockmean-normmean
base=base.where(shock.rolling(60,min_periods=45).sum()>=7, np.nan)
# residualize cross-sectionally vs contemporaneously known trailing trend and volatility
mom=ret.rolling(20,min_periods=15).sum(); vol=ret.rolling(20,min_periods=15).std()
sig=pd.DataFrame(index=close.index,columns=A,dtype=float)
for dt in close.index:
    y=base.loc[dt]; X=pd.DataFrame({'m':mom.loc[dt], 'v':vol.loc[dt]})
    z=pd.concat([y.rename('y'),X],axis=1).dropna()
    if len(z)>=8:
        xx=np.column_stack([np.ones(len(z)),z[['m','v']].values])
        sig.loc[dt,z.index]=z.y.values-xx@np.linalg.lstsq(xx,z.y.values,rcond=None)[0]
# Factor is available after close t and evaluated next forward holding return.
def csic(h):
    fwd=close.shift(-h)/close-1
    vals=[]
    for dt in sig.index:
        q=pd.concat([sig.loc[dt].rename('s'),fwd.loc[dt].rename('r')],axis=1).dropna()
        if len(q)>=8: vals.append(q.s.corr(q.r,method='spearman'))
    x=np.asarray(vals,float); return len(x),float(np.mean(x)),float(np.mean(x)/np.std(x,ddof=1)) if len(x)>1 and np.std(x,ddof=1)>0 else np.nan,float(np.mean(x>0))
print('FACTOR: residual_vix_shock_conditional_resilience_60_20')
print('cutoff',close.index.max().date(),'assets',len(A),'factor_dates',int(sig.notna().any(axis=1).sum()),'valid_cells',int(sig.notna().sum()),'coverage',round(sig.notna().mean().mean(),6),'mean_names',round(sig.notna().sum(axis=1).mean(),3))
for h in [1,5,10,20]: print('H',h,'n IC ICIR hit',csic(h))
# rank turnover and concentration
rk=sig.rank(axis=1,pct=True); print('turnover',float(rk.diff().abs().mean(axis=1).mean()),'concentration',float(((sig-sig.mean(axis=1).values[:,None])/sig.std(axis=1).values[:,None]).abs().stack().mean()))
for lo,hi,name in [('2020-01-01','2024-12-31','2020-24'),('2025-01-01','2029-12-31','2025-29'),('2030-01-01','2034-12-31','2030-34'),('2035-01-01','2100-01-01','2035YTD')]:
    old=sig.copy(); sig=sig.loc[lo:hi]; print('REGIME',name,'H5',csic(5)); sig=old
# library evidence: signal panels are conventionally pkl alongside research scripts
panels=glob.glob('scripts/*signal.pkl'); corrs=[]; compared=[]
for p in panels:
    try:
        other=pd.read_pickle(p)
        if isinstance(other,pd.DataFrame):
            q=pd.concat([sig.stack().rename('a'),other.stack().rename('b')],axis=1).dropna()
            if len(q)>=100: corrs.append(abs(q.a.corr(q.b,method='spearman'))); compared.append((os.path.basename(p),len(q),corrs[-1]))
    except Exception: pass
print('LIBRARY_PANELS',len(compared),'MAXCORR',max(corrs) if corrs else None,'DETAIL',sorted(compared,key=lambda x:-x[2])[:5])
sig.to_pickle('scripts/miner_1_20350816_residual_vix_shock_conditional_resilience_60_20_signal.pkl')
