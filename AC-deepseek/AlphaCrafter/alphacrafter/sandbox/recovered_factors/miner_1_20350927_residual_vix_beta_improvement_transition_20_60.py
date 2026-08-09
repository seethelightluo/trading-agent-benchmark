import os,json,glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
# One interpretable idea: macro sensitivity transition.  A high score means an
# asset's return beta to daily VIX changes has become less adverse over 20 days
# versus its own 60-day baseline, net of ordinary trend and realized volatility.
END=5000
acc=get_account_dict(); A=list(acc['watch_list'])
def load(a):
 x=get_stock_daily_data(a,END).copy(); x['date']=pd.to_datetime(x.date); return x.set_index('date').sort_index()
raw={a:load(a) for a in A}; close=pd.DataFrame({a:x.close for a,x in raw.items()}).sort_index(); ret=close.pct_change()
v=get_index_daily_data('VIX',END).copy(); v['date']=pd.to_datetime(v.date); v=v.set_index('date').sort_index(); vc='close' if 'close' in v else v.select_dtypes('number').columns[0]
dv=v[vc].pct_change().reindex(close.index).ffill()
# rolling beta uses only returns and macro observations through each signal date
var20=dv.rolling(20,min_periods=16).var(); var60=dv.rolling(60,min_periods=48).var()
b20=ret.rolling(20,min_periods=16).cov(dv).div(var20,axis=0)
b60=ret.rolling(60,min_periods=48).cov(dv).div(var60,axis=0)
base=b20-b60
mom=ret.rolling(20,min_periods=16).sum(); vol=ret.rolling(20,min_periods=16).std()
sig=pd.DataFrame(index=close.index,columns=A,dtype=float)
for dt in sig.index:
 z=pd.concat([base.loc[dt].rename('y'),mom.loc[dt].rename('m'),vol.loc[dt].rename('v')],axis=1).dropna()
 if len(z)>=8:
  X=np.column_stack([np.ones(len(z)),z[['m','v']].values]); sig.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y.values,rcond=None)[0]
def metrics(panel,h,sub=None):
 fwd=close.shift(-h).div(close)-1; vals=[]
 ix=panel.index if sub is None else panel.loc[sub[0]:sub[1]].index
 for dt in ix:
  q=pd.concat([panel.loc[dt].rename('s'),fwd.loc[dt].rename('r')],axis=1).dropna()
  if len(q)>=8: vals.append(q.s.corr(q.r,method='spearman'))
 x=np.array(vals,float); return len(x),x.mean() if len(x) else np.nan,x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan,(x>0).mean() if len(x) else np.nan
print('FACTOR residual_vix_beta_improvement_transition_20_60')
print('cutoff',close.index.max().date(),'assets',len(A),'factor_dates',sig.notna().any(axis=1).sum(),'valid_cells',sig.notna().sum().sum(),'coverage',sig.notna().mean().mean(),'mean_names',sig.notna().sum(axis=1).mean())
for h in (1,5,10,20): print('H',h,'n IC ICIR hit',metrics(sig,h))
r=sig.rank(axis=1,pct=True); z=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1),axis=0); print('turnover',r.diff().abs().mean(axis=1).mean(),'concentration',z.abs().stack().mean())
for lo,hi,n in [('2020-01-01','2024-12-31','2020-24'),('2025-01-01','2029-12-31','2025-29'),('2030-01-01','2034-12-31','2030-34'),('2035-01-01','2100-01-01','2035YTD')]: print('REGIME',n,'H5',metrics(sig,5,(lo,hi)))
# Independence evidence only against active admitted JSONs, resolving their signal panels by date/id fragments.
active=[]
for f in glob.glob('factors/*.json'):
 try:
  d=json.load(open(f));
  if d.get('validation',{}).get('status')=='EFFECTIVE': active.append((os.path.basename(f),d.get('factor_id','')))
 except: pass
corr=[]
for fn,fid in active:
 hits=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not hits: continue
 try:
  o=pd.read_pickle(hits[-1]); q=pd.concat([sig.stack().rename('a'),o.stack().rename('b')],axis=1).dropna()
  if len(q)>=100: corr.append((abs(q.a.corr(q.b,method='spearman')),fn,len(q)))
 except: pass
print('ACTIVE_FACTORS',len(active),'CORRELATION_EVIDENCE',len(corr),'MAXCORR',max(corr) if corr else None,'TOP',sorted(corr,reverse=True)[:5])
sig.to_pickle('scripts/miner_1_20350927_residual_vix_beta_improvement_transition_20_60_signal.pkl')
