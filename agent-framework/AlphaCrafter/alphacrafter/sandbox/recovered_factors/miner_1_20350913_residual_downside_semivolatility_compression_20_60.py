import os, glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

# One interpretable idea: assets with a recent reduction in downside semivolatility
# relative to their own medium-term baseline may be emerging from stress.  Remove
# ordinary total-volatility and trailing-return levels cross-sectionally.
END=5000
acc=get_account_dict(); A=list(acc['watch_list'])
def load(a):
    x=get_stock_daily_data(a,END).copy(); x['date']=pd.to_datetime(x.date)
    return x.set_index('date').sort_index()
raw={a:load(a) for a in A}
close=pd.DataFrame({a:x.close for a,x in raw.items()}).sort_index(); ret=close.pct_change()
neg=ret.where(ret<0,0.0)
dsv20=np.sqrt((neg**2).rolling(20,min_periods=15).mean())
dsv60=np.sqrt((neg**2).rolling(60,min_periods=45).mean())
# Positive = downside risk has compressed versus its own baseline.
base=-(dsv20/dsv60-1.0)
mom=ret.rolling(20,min_periods=15).sum(); vol=ret.rolling(20,min_periods=15).std()
sig=pd.DataFrame(np.nan,index=close.index,columns=A)
for dt in close.index:
    z=pd.concat([base.loc[dt].rename('y'),mom.loc[dt].rename('m'),vol.loc[dt].rename('v')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)>=8:
        X=np.column_stack([np.ones(len(z)),z[['m','v']].values])
        sig.loc[dt,z.index]=z.y.values-X@np.linalg.lstsq(X,z.y.values,rcond=None)[0]
def met(panel,h,ix=None):
    fw=close.shift(-h).div(close)-1; vals=[]
    dates=panel.index if ix is None else panel.loc[ix].index
    for dt in dates:
        q=pd.concat([panel.loc[dt].rename('s'),fw.loc[dt].rename('r')],axis=1).dropna()
        if len(q)>=8: vals.append(q.s.corr(q.r,method='spearman'))
    x=np.asarray(vals,float); sd=x.std(ddof=1) if len(x)>1 else np.nan
    return len(x),float(x.mean()),float(x.mean()/sd) if sd>0 else np.nan,float((x>0).mean())
print('FACTOR residual_downside_semivolatility_compression_20_60')
print('cutoff',close.index.max().date(),'assets',len(A),'factor_dates',int(sig.notna().any(axis=1).sum()),'valid_cells',int(sig.notna().sum().sum()),'coverage',round(float(sig.notna().mean().mean()),6),'mean_names',round(float(sig.notna().sum(axis=1).mean()),3))
for h in (1,5,10,20): print('H',h,'n IC ICIR hit',met(sig,h))
ranks=sig.rank(axis=1,pct=True); z=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1),axis=0)
print('turnover',float(ranks.diff().abs().mean(axis=1).mean()),'concentration',float(z.abs().stack().mean()))
for lo,hi,n in [('2020-01-01','2024-12-31','2020-24'),('2025-01-01','2029-12-31','2025-29'),('2030-01-01','2034-12-31','2030-34'),('2035-01-01','2100-01-01','2035YTD')]: print('REGIME',n,'H5',met(sig,5,slice(lo,hi)))
# Broad scan is deliberately stricter than the admitted-library comparison: a
# candidate can pass only if it is independent even of every available research panel.
detail=[]
for p in glob.glob('scripts/*signal.pkl'):
 try:
  o=pd.read_pickle(p)
  if not isinstance(o,pd.DataFrame): continue
  q=pd.concat([sig.stack().rename('a'),o.stack().rename('b')],axis=1).dropna()
  if len(q)>=100 and q.a.nunique()>1 and q.b.nunique()>1:
   rho=q.a.corr(q.b,method='spearman')
   if np.isfinite(rho): detail.append((os.path.basename(p),len(q),abs(float(rho))))
 except Exception: pass
detail.sort(key=lambda x:-x[2])
print('LIBRARY_PANELS',len(detail),'MAXCORR',detail[0] if detail else None,'TOP5',detail[:5])
sig.to_pickle('scripts/miner_1_20350913_residual_downside_semivolatility_compression_20_60_signal.pkl')
