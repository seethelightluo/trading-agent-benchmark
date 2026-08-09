"""One idea: return-path upside dominance (20 days).
This price-only factor measures whether an asset's recent absolute return path
has been dominated by positive rather than negative daily moves.  Unlike total
momentum, it distinguishes equal endpoint returns reached with asymmetric
up/down participation.  Higher values denote positive path dominance.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-08-29')
def load(a):
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
    return x.loc[:CUT,['close']]
dat={a:load(a) for a in A}
end=min(x.close.dropna().index.max() for x in dat.values())
idx=pd.date_range(min(x.index.min() for x in dat.values()),end,freq='B')
close=pd.DataFrame({a:x.close.reindex(idx).ffill() for a,x in dat.items()})
r=close.pct_change().clip(-.25,.25)
pos=r.clip(lower=0).rolling(20,min_periods=15).sum()
neg=(-r.clip(upper=0)).rolling(20,min_periods=15).sum()
f=(pos-neg).div((pos+neg).replace(0,np.nan)).clip(-1,1)
print('FACTOR return_path_upside_dominance_20 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 fw=close.shift(-h).div(close)-1; vals=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v): vals.append((d,v));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01','2034-12-31'),('2035','2035-01-01',end)]:
 s=ics[20].loc[lo:hi]
 print('REGIME20 %s dates=%d IC=%s ICIR=%s hit=%s'%(nm,len(s),('%.6f'%s.mean() if len(s) else 'N/A'),('%.6f'%(s.mean()/s.std(ddof=1)) if len(s)>1 else 'N/A'),('%.6f'%(s>0).mean() if len(s) else 'N/A')))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: tos.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(tos),len(tos),z.abs().stack().mean()))
effids=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE':effids.append(j['factor_id'])
 except Exception: pass
scores=[];missing=[]
for fid in effids:
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths: missing.append(fid);continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime)); old=old.get('signal',old) if isinstance(old,dict) else old
 q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2:missing.append(fid);continue
 scores.append(abs(spearmanr(q.x,q.z).statistic))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(effids),len(scores),len(missing),('%.6f'%max(scores) if len(scores)==len(effids) else 'UNAVAILABLE')))
f.to_pickle('scripts/miner_3_20350830_return_path_upside_dominance_20_signal.pkl')
