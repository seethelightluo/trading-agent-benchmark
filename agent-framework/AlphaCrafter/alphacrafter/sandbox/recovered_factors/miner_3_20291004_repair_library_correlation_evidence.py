"""Repair all-library correlation evidence for inverse peer-dispersion sensitivity.
Only evaluates the existing candidate; corrects time-axis multiplication in stress signal
and uses flexible macro retrieval to diagnose DXY alignment."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']
def series(sym, macro=False):
    loaders=[get_index_daily_data,get_stock_daily_data] if macro else [get_stock_daily_data]
    for fn in loaders:
        try:
            d=fn(sym,5000).copy()
            if len(d) and 'close' in d:
                d['date']=pd.to_datetime(d['date']); x=pd.to_numeric(d.set_index('date')['close'],errors='coerce').sort_index()
                if x.notna().any(): return x
        except Exception as e: print('LOAD_FAIL',sym,fn.__name__,type(e).__name__)
    return pd.Series(dtype=float)
P=pd.DataFrame({a:series(a) for a in A}).sort_index(); r=P.pct_change(); m=r.median(axis=1); rel=r.sub(m,axis=0)
cs=lambda x:x.sub(x.median(axis=1),axis=0)
disp=rel.abs().median(axis=1)
F=cs(pd.DataFrame({a:-rel[a].abs().rolling(60,min_periods=40).corr(disp) for a in A}).shift(1))
# exact admitted stress definition: row-wise stress weight multiplication
stress=m < -0.35*m.rolling(60,min_periods=30).std().shift(1)
weight=1+0.25*stress.rolling(5,min_periods=1).sum().shift(1)
S=cs(-rel.where(stress,axis=0).mul(weight,axis=0).rolling(60,min_periods=5).mean().shift(1))
q=pd.concat([F.stack(),S.stack()],axis=1).dropna()
print('CUTOFF',P.index.max().date(),'CANDIDATE_CELLS',int(F.notna().sum().sum()))
print('STRESS', 'cells',len(q),'stress_share',round(float(stress.mean()),6),'signal_cells',int(S.notna().sum().sum()),'rho',round(float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic),6))
for sym in ['DXY','VIX']:
 x=series(sym,True); print('MACRO',sym,'n',len(x),'nonnull',int(x.notna().sum()),'start',x.index.min() if len(x) else None,'end',x.index.max() if len(x) else None,'overlap',int(x.reindex(P.index).notna().sum()))
# Direct DXY formula conditionally returns cross-sectional nonmissing values only if macro overlap exists.
D=series('DXY',True).reindex(P.index).pct_change()
G=pd.DataFrame({a:r[a].where(D<0).rolling(60,min_periods=35).mean()-r[a].where(D>0).rolling(60,min_periods=35).mean() for a in A})
q2=pd.concat([F.stack(),G.stack()],axis=1).dropna(); print('DXY_SIGNAL','cells',len(q2),'signal_cells',int(G.notna().sum().sum()),'rho',round(float(spearmanr(q2.iloc[:,0],q2.iloc[:,1]).statistic),6) if len(q2)>8 else None)
