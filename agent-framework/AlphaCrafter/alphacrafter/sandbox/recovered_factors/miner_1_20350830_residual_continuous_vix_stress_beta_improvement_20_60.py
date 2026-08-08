import os, glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

# One idea: residualized continuous VIX-stress beta transition.  Score assets by
# the reduction in their rolling return sensitivity to standardized VIX changes
# from 60d baseline to recent 20d, then remove ordinary return/volatility levels.
END=5000
acc=get_account_dict(); A=list(acc['watch_list'])
def asset(a):
    x=get_stock_daily_data(a,END).copy(); x['date']=pd.to_datetime(x.date)
    return x.set_index('date').sort_index()
raw={a:asset(a) for a in A}
close=pd.DataFrame({a:x.close for a,x in raw.items()}).sort_index()
r=close.pct_change()
v=get_index_daily_data('VIX',END).copy(); v['date']=pd.to_datetime(v.date); v=v.set_index('date').sort_index()
vc='close' if 'close' in v else v.select_dtypes('number').columns[0]
vr=v[vc].pct_change().reindex(close.index).ffill()
# VIX return standardized using only contemporaneous/past information
zv=(vr-vr.rolling(60,min_periods=45).mean())/vr.rolling(60,min_periods=45).std()
def beta(w):
    return r.rolling(w,min_periods=int(w*.75)).cov(zv).div(zv.rolling(w,min_periods=int(w*.75)).var(),axis=0)
b20,b60=beta(20),beta(60)
base=-(b20-b60) # high means sensitivity to stress has improved (fallen)
mom=r.rolling(20,min_periods=15).sum(); vol=r.rolling(20,min_periods=15).std()
sig=pd.DataFrame(np.nan,index=close.index,columns=A)
for dt in close.index:
    z=pd.concat([base.loc[dt].rename('y'),mom.loc[dt].rename('m'),vol.loc[dt].rename('v')],axis=1).dropna()
    if len(z)>=8:
        X=np.column_stack([np.ones(len(z)),z[['m','v']]])
        sig.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def metrics(panel,h,ix=None):
    fw=close.shift(-h)/close-1; vals=[]
    for dt in (panel.index if ix is None else panel.loc[ix].index):
        q=pd.concat([panel.loc[dt].rename('s'),fw.loc[dt].rename('r')],axis=1).dropna()
        if len(q)>=8: vals.append(q.s.corr(q.r,method='spearman'))
    x=np.array(vals,float); sd=np.std(x,ddof=1) if len(x)>1 else np.nan
    return len(x),np.mean(x),np.mean(x)/sd if sd and sd>0 else np.nan,np.mean(x>0)
print('FACTOR residual_continuous_vix_stress_beta_improvement_20_60')
print('cutoff',close.index.max().date(),'assets',len(A),'factor_dates',int(sig.notna().any(axis=1).sum()),'valid_cells',int(sig.notna().sum().sum()),'coverage',round(sig.notna().mean().mean(),6),'mean_names',round(sig.notna().sum(axis=1).mean(),3))
for h in (1,5,10,20): print('H',h,'n IC ICIR hit',metrics(sig,h))
ranks=sig.rank(axis=1,pct=True)
print('turnover',ranks.diff().abs().mean(axis=1).mean(),'concentration',((sig.sub(sig.mean(axis=1),axis=0)).div(sig.std(axis=1),axis=0)).abs().stack().mean())
for lo,hi,n in [('2020-01-01','2024-12-31','2020-24'),('2025-01-01','2029-12-31','2025-29'),('2030-01-01','2034-12-31','2030-34'),('2035-01-01','2100-01-01','2035YTD')]: print('REGIME',n,'H5',metrics(sig,5,slice(lo,hi)))
c=[]; detail=[]
for p in glob.glob('scripts/*signal.pkl'):
 try:
  o=pd.read_pickle(p)
  if isinstance(o,pd.DataFrame):
   q=pd.concat([sig.stack().rename('a'),o.stack().rename('b')],axis=1).dropna()
   if len(q)>=100:
    rho=abs(q.a.corr(q.b,method='spearman')); c.append(rho); detail.append((os.path.basename(p),len(q),rho))
 except Exception: pass
print('LIBRARY_PANELS',len(c),'MAXCORR',max(c) if c else None,'DETAIL',sorted(detail,key=lambda x:-x[2])[:5])
sig.to_pickle('scripts/miner_1_20350830_residual_continuous_vix_stress_beta_improvement_20_60_signal.pkl')
