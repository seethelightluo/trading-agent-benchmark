"""One interpretable idea: residual short-horizon reversal.
At each completed date, use the NEGATIVE residual of 5-day return after a
cross-sectional linear regression on 60-day return and 20-day realised
volatility.  It tests whether unusually sharp moves, beyond an asset's medium
trend and risk, mean-revert over subsequent horizons.  No forward information
is used in the signal."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-03-28')
def load(a):
    return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:CUT]
raw=pd.DataFrame({a:load(a) for a in A}); end=raw.dropna(how='any').index.max()
p=raw.loc[:end].reindex(pd.date_range(raw.index.min(),end,freq='B')).ffill(); r=p.pct_change()
r5=p.pct_change(5); r60=p.pct_change(60); vol20=r.rolling(20,min_periods=20).std()
f=pd.DataFrame(np.nan,index=p.index,columns=A)
for d in p.index:
    q=pd.concat([r5.loc[d].rename('r5'),r60.loc[d].rename('r60'),vol20.loc[d].rename('vol')],axis=1).dropna()
    if len(q)>=8 and q[['r5','r60','vol']].nunique().min()>1:
        x=np.column_stack([np.ones(len(q)),q.r60.values,q.vol.values])
        f.loc[d,q.index]=-(q.r5.values-x@np.linalg.lstsq(x,q.r5.values,rcond=None)[0])
print('FACTOR residual_short_horizon_reversal_5_60_20 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v): vals.append((d,v));ns.append(len(q))
 s=pd.Series(dict(vals));ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01','2034-12-31'),('2035','2035-01-01',end)]:
 s=ics[5].loc[lo:hi]
 print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(ts),len(ts),z.abs().stack().mean()))
effids=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE':effids.append(j['factor_id'])
 except Exception: pass
scores=[];missing=[];peers=[]
for fid in effids:
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths: missing.append(fid);continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime)); old=old.get('signal',old) if isinstance(old,dict) else old
 q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2:missing.append(fid);continue
 rho=abs(spearmanr(q.x,q.z).statistic);scores.append(rho);peers.append((rho,fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(effids),len(scores),len(missing),('%.6f'%max(scores) if len(scores)==len(effids) and scores else 'UNAVAILABLE')))
if peers: print('MAX_OBSERVED rho=%.6f factor=%s cells=%d'%max(peers))
f.to_pickle('scripts/miner_1_20350329_residual_short_horizon_reversal_5_60_20_signal.pkl')
