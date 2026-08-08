import os, glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

# Single idea: relative drawdown-recovery acceleration.  A positive value means
# an asset has repaired more of its trailing 60d peak-to-trough loss in the last
# 20 days than peers, beyond ordinary 20/60d returns and volatility.
N=5000
acc=get_account_dict(); A=list(acc['watch_list'])
def load(a):
    x=get_stock_daily_data(a,N).copy(); x['date']=pd.to_datetime(x['date'])
    return x.set_index('date').sort_index()
raw={a:load(a) for a in A}
close=pd.DataFrame({a:x['close'] for a,x in raw.items()}).sort_index()
r=close.pct_change(); m20=close.pct_change(20); m60=close.pct_change(60); v20=r.rolling(20,min_periods=15).std()
# Drawdown is non-positive. Its 20d change measures speed of repair; use a
# lagged 60d rolling high throughout, so no forward data enter the score.
dd=close.div(close.rolling(60,min_periods=40).max())-1
base=dd-dd.shift(20)
sig=pd.DataFrame(np.nan,index=close.index,columns=A)
for d in close.index:
    q=pd.concat([base.loc[d].rename('y'),m20.loc[d].rename('m20'),m60.loc[d].rename('m60'),v20.loc[d].rename('v')],axis=1).dropna()
    if len(q)>=8:
        X=np.column_stack([np.ones(len(q)),q[['m20','m60','v']].to_numpy()])
        sig.loc[d,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
def met(p,h,sel=None):
    fw=close.shift(-h).div(close)-1; vals=[]
    ix=p.loc[sel].index if sel is not None else p.index
    for d in ix:
        q=pd.concat([p.loc[d].rename('s'),fw.loc[d].rename('f')],axis=1).dropna()
        if len(q)>=8 and q.s.nunique()>1 and q.f.nunique()>1: vals.append(q.s.corr(q.f,method='spearman'))
    x=np.array(vals); sd=x.std(ddof=1) if len(x)>1 else np.nan
    return len(x),float(x.mean()),float(x.mean()/sd),float((x>0).mean())
print('FACTOR residual_drawdown_recovery_acceleration_20_60')
print('cutoff',close.index.max().date(),'assets',len(A),'signal_dates',int(sig.notna().any(axis=1).sum()),'cells',int(sig.notna().sum().sum()),'coverage',float(sig.notna().mean().mean()),'mean_names',float(sig.notna().sum(axis=1).mean()))
for h in (1,5,10,20): print('H',h,'n_IC_ICIR_hit',met(sig,h))
rnk=sig.rank(axis=1,pct=True); z=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1),axis=0)
print('turnover',float(rnk.diff().abs().mean(axis=1).mean()),'concentration',float(z.abs().stack().mean()))
for lo,hi,label in [('2020-01-01','2024-12-31','2020_24'),('2025-01-01','2029-12-31','2025_29'),('2030-01-01','2034-12-31','2030_34'),('2035-01-01','2100-01-01','2035YTD')]: print('REGIME',label,'H10',met(sig,10,slice(lo,hi)))
rows=[]
for p in glob.glob('scripts/*signal.pkl'):
 try:
  o=pd.read_pickle(p)
  if isinstance(o,pd.DataFrame):
   q=pd.concat([sig.stack().rename('a'),o.stack().rename('b')],axis=1).dropna()
   if len(q)>=100 and q.a.nunique()>1 and q.b.nunique()>1: rows.append((os.path.basename(p),len(q),abs(float(q.a.corr(q.b,method='spearman')))))
 except Exception: pass
rows.sort(key=lambda x:-x[2]); print('PANEL_COMPARISONS',len(rows),'MAX',rows[0] if rows else None,'TOP5',rows[:5])
sig.to_pickle('scripts/miner_1_20351122_residual_drawdown_recovery_acceleration_20_60_signal.pkl')
