"""One candidate: residual momentum acceleration, cross-asset full-coverage validation."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2034-01-18')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def close(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 return d.loc[:CUT,'close']
p=pd.DataFrame({a:close(a) for a in A}); r=p.pct_change(); res=r.sub(r.mean(axis=1),axis=0)
# Recent residual trend relative to its preceding equal-length residual trend.
# Positive value means an idiosyncratic trend has accelerated rather than merely being high momentum.
f=res.rolling(20,min_periods=20).sum()-res.shift(20).rolling(20,min_periods=20).sum()
print('FACTOR residual_momentum_acceleration_20_20 VALIDATED_THROUGH',CUT.date())
print('definition=sum(residual_return,t-19:t)-sum(residual_return,t-39:t-20); residual=asset return-equal-weight universe return')
print('coverage=%.6f valid_dates=%d valid_cells=%d assets=%d'%(f.notna().mean().mean(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),len(A)))
stats={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v): vals.append((d,v));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float);stats[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for label,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2033','2027-01-01','2033-12-31'),('2034_ytd','2034-01-01',CUT)]:
 s=stats[5].loc[lo:hi]
 print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(label,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('rank_turnover=%.6f pairs=%d'%(np.mean(ts),len(ts)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn))
  if x.get('validation',{}).get('status')=='EFFECTIVE':eff.append(x['factor_id'])
 except Exception:pass
scores=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:missing.append(fid);continue
 z=pd.read_pickle(max(hits,key=os.path.getmtime))
 q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1:scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d artifacts=%d missing=%d'%(len(eff),len(scores),len(missing)))
if scores: print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
if missing:print('MISSING_ARTIFACTS='+','.join(missing))
f.to_pickle('scripts/miner_2_20340119_residual_momentum_acceleration_20_20_signal.pkl')
