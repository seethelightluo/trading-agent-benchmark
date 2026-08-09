"""One candidate: low-dispersion-conditioned residual directional efficiency (20 sessions)."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def close(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
def unwrap(x):
 if isinstance(x,dict):
  for k in ('signal','factor','data','values'):
   if k in x: x=x[k]; break
 if isinstance(x,pd.Series): return x.unstack() if isinstance(x.index,pd.MultiIndex) else x.to_frame()
 return x if isinstance(x,pd.DataFrame) else None
p=pd.DataFrame({a:close(a) for a in A})
CUT=min(p.dropna(how='all').index.max(),pd.Timestamp('2035-01-31')); p=p.loc[:CUT]
r=p.pct_change(); resid=r.sub(r.mean(axis=1),axis=0)
base=resid.rolling(20,min_periods=15).sum().shift(1)/(resid.abs().rolling(20,min_periods=15).sum().shift(1)+1e-12)
# At t use only completed observations: current cross-asset residual dispersion is lagged,
# and the threshold is its prior 60-session median.  Signal is active only in quiet,
# low-dispersion regimes, where efficient idiosyncratic leadership should persist.
disp=resid.std(axis=1).shift(1)
quiet=disp < disp.rolling(60,min_periods=40).median().shift(1)
f=base.where(quiet,axis=0)
print('FACTOR low_dispersion_conditioned_residual_directional_efficiency_20 VALIDATED_THROUGH',CUT.date())
print('definition=lagged 20d residual directional efficiency, active only when lagged cross-asset residual dispersion is below its lagged 60d median')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f active_dates=%d'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),quiet.sum()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; obs=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): obs.append((d,z));ns.append(len(q))
 s=pd.Series(dict(obs),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01','2033-12-31'),('2034_35','2034-01-01',CUT)]:
 s=ics[10].loc[lo:hi]
 print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turns),len(turns)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE': eff.append(j['factor_id'])
 except: pass
scores=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: missing.append(fid);continue
 z=unwrap(pd.read_pickle(max(hits,key=os.path.getmtime)))
 if z is None:missing.append(fid);continue
 q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if scores: print('largest_evidenced='+str(max(scores)))
print('MAX_ABS_LIBRARY_CORRELATION='+('%.6f'%max(scores)[0] if len(scores)==len(eff) else 'UNAVAILABLE'))
f.to_pickle('scripts/miner_2_20350201_low_dispersion_conditioned_residual_directional_efficiency_20_signal.pkl')
