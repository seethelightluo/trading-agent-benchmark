import os, glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

# Single price-only idea: medium-term trend-conditioned short-term reversal.
# A 5-day pullback is ranked positively only when the asset remains above its
# own 60-day trend; in downtrends signal is set to zero rather than extrapolating
# reversal across fundamentally different regimes.
N=5000
acc=get_account_dict(); A=list(acc['watch_list'])
def load(a):
    x=get_stock_daily_data(a,N).copy(); x['date']=pd.to_datetime(x['date'])
    return x.set_index('date').sort_index()
raw={a:load(a) for a in A}
close=pd.DataFrame({a:x['close'] for a,x in raw.items()}).sort_index()
r5=close.pct_change(5); r60=close.pct_change(60)
sig=(-r5).where(r60>0,0.0)
# Cross-sectional residualization against the level of 60d trend ensures the
# test is pullback severity, rather than merely loading on stronger trends.
for dt in sig.index:
    q=pd.concat([sig.loc[dt].rename('y'),r60.loc[dt].rename('m')],axis=1).dropna()
    if len(q)>=8:
        X=np.column_stack([np.ones(len(q)),q.m.to_numpy()])
        sig.loc[dt,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
    else: sig.loc[dt]=np.nan
def metrics(panel,h,sel=None):
    fw=close.shift(-h).div(close)-1; out=[]
    dates=panel.loc[sel].index if sel is not None else panel.index
    for d in dates:
        q=pd.concat([panel.loc[d].rename('s'),fw.loc[d].rename('r')],axis=1).dropna()
        if len(q)>=8 and q.s.nunique()>1 and q.r.nunique()>1: out.append(q.s.corr(q.r,method='spearman'))
    x=np.asarray(out,float); sd=x.std(ddof=1) if len(x)>1 else np.nan
    return len(x),float(x.mean()) if len(x) else np.nan,float(x.mean()/sd) if sd and sd>0 else np.nan,float((x>0).mean()) if len(x) else np.nan
print('FACTOR trend_conditioned_pullback_reversal_5_60')
print('cutoff',close.index.max().date(),'assets',len(A),'signal_dates',int(sig.notna().any(axis=1).sum()),'cells',int(sig.notna().sum().sum()),'coverage',float(sig.notna().mean().mean()),'mean_names',float(sig.notna().sum(axis=1).mean()))
for h in (1,5,10,20): print('H',h,'n_IC_ICIR_hit',metrics(sig,h))
r=sig.rank(axis=1,pct=True); z=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1),axis=0)
print('turnover',float(r.diff().abs().mean(axis=1).mean()),'concentration',float(z.abs().stack().mean()))
for lo,hi,label in [('2020-01-01','2024-12-31','2020_24'),('2025-01-01','2029-12-31','2025_29'),('2030-01-01','2034-12-31','2030_34'),('2035-01-01','2100-01-01','2035YTD')]: print('REGIME',label,'H5',metrics(sig,5,slice(lo,hi)))
rows=[]
for p in glob.glob('scripts/*signal.pkl'):
    try:
        o=pd.read_pickle(p)
        if not isinstance(o,pd.DataFrame): continue
        q=pd.concat([sig.stack().rename('a'),o.stack().rename('b')],axis=1).dropna()
        if len(q)>=100 and q.a.nunique()>1 and q.b.nunique()>1:
            rho=q.a.corr(q.b,method='spearman')
            if np.isfinite(rho): rows.append((os.path.basename(p),len(q),abs(float(rho))))
    except Exception: pass
rows.sort(key=lambda x:-x[2]); print('PANEL_COMPARISONS',len(rows),'MAX',rows[0] if rows else None,'TOP5',rows[:5])
sig.to_pickle('scripts/miner_1_20351108_trend_conditioned_pullback_reversal_5_60_signal.pkl')
