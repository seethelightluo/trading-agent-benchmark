import os, glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

# Single idea: abnormal-volume confirmation of a medium-term price breakout.
# Per-asset volume is normalized to its own 60-day median, then confirmation
# is signed by 20-day return. Cross-sectionally remove ordinary momentum and
# total volatility, testing whether unusual participation adds distinct content.
N=5000
acc=get_account_dict(); A=list(acc['watch_list'])
def load(a):
    x=get_stock_daily_data(a,N).copy(); x['date']=pd.to_datetime(x['date'])
    return x.set_index('date').sort_index()
raw={a:load(a) for a in A}
close=pd.DataFrame({a:x['close'] for a,x in raw.items()}).sort_index()
volume=pd.DataFrame({a:x['volume'] if 'volume' in x else np.nan for a,x in raw.items()}).reindex(close.index)
ret=close.pct_change(); mom=ret.rolling(20,min_periods=15).sum(); vol=ret.rolling(20,min_periods=15).std()
med=volume.rolling(60,min_periods=40).median()
# log ratio limits scale effects; sign ties participation to direction of the prior move.
part=np.log(volume/med).replace([np.inf,-np.inf],np.nan)
base=part*np.sign(mom)
sig=pd.DataFrame(np.nan,index=close.index,columns=A)
for dt in close.index:
    q=pd.concat([base.loc[dt].rename('y'),mom.loc[dt].rename('m'),vol.loc[dt].rename('v')],axis=1).dropna()
    if len(q)>=8:
        X=np.column_stack([np.ones(len(q)),q[['m','v']].to_numpy()])
        sig.loc[dt,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
def metrics(panel,h,sel=None):
    fw=close.shift(-h).div(close)-1; out=[]
    dates=panel.loc[sel].index if sel is not None else panel.index
    for d in dates:
        q=pd.concat([panel.loc[d].rename('s'),fw.loc[d].rename('r')],axis=1).dropna()
        if len(q)>=8 and q.s.nunique()>1 and q.r.nunique()>1: out.append(q.s.corr(q.r,method='spearman'))
    x=np.array(out,float); sd=x.std(ddof=1) if len(x)>1 else np.nan
    return (len(x),float(x.mean()) if len(x) else np.nan,float(x.mean()/sd) if sd and sd>0 else np.nan,float((x>0).mean()) if len(x) else np.nan)
print('FACTOR residual_volume_confirmed_breakout_20_60')
print('cutoff',close.index.max().date(),'assets',len(A),'signal_dates',int(sig.notna().any(axis=1).sum()),'cells',int(sig.notna().sum().sum()),'coverage',float(sig.notna().mean().mean()),'mean_names',float(sig.notna().sum(axis=1).mean()))
for h in (1,5,10,20): print('H',h,'n_IC_ICIR_hit',metrics(sig,h))
r=sig.rank(axis=1,pct=True); z=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1),axis=0)
print('turnover',float(r.diff().abs().mean(axis=1).mean()),'concentration',float(z.abs().stack().mean()))
for lo,hi,label in [('2020-01-01','2024-12-31','2020_24'),('2025-01-01','2029-12-31','2025_29'),('2030-01-01','2034-12-31','2030_34'),('2035-01-01','2100-01-01','2035YTD')]: print('REGIME',label,'H5',metrics(sig,5,slice(lo,hi)))
# Compare all recoverable research panels; this is evidence, but an admission
# additionally requires admitted-factor provenance to be resolvable.
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
sig.to_pickle('scripts/miner_1_20351025_residual_volume_confirmed_breakout_20_60_signal.pkl')
