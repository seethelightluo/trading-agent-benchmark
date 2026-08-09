"""One candidate: dispersion-conditioned cross-sectional short-term reversal, using bars completed through 2033-12-07."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CUT = pd.Timestamp('2033-12-07')
A = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p = pd.DataFrame({a: pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r = p.pct_change()
# Cross-asset dispersion measures how unusually differentiated the prior day was.
disp = r.std(axis=1, ddof=0)
state = disp > disp.rolling(60, min_periods=40).quantile(.75)
# Only activate reversal after high dispersion; neutral (zero) otherwise so every asset retains coverage.
f = (-p.pct_change(5)).where(state, 0.0).replace([np.inf,-np.inf],np.nan)
print('FACTOR dispersion_conditioned_short_reversal_5d VALIDATED_THROUGH', CUT.date())
print('definition=if cross-sectional daily-return dispersion > trailing-60d 75th percentile: negative trailing 5d return; else 0')
print('state_frequency=%.4f coverage=%.6f valid_dates=%d valid_cells=%d assets=%d' % (state.mean(), f.notna().mean().mean(), f.notna().any(axis=1).sum(), f.notna().sum().sum(), len(A)))
all_stats={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v): vals.append((d,v));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float); ir=s.mean()/s.std(ddof=1)
 all_stats[h]=(s.mean(),ir,len(s))
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),ir,len(s),(s>0).mean(),np.mean(ns)))
 if h==5:
  for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01',CUT)]:
   z=s.loc[lo:hi]
   print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(n,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turns),len(turns)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE': eff.append(z['factor_id'])
 except Exception: pass
found=[]; scores=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if hits:
  found.append(fid); z=pd.read_pickle(max(hits,key=os.path.getmtime))
  q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1: scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE artifacts=%d effective=%d'%(len(found),len(eff)))
if scores: print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
print('ADMISSION=FAIL if artifacts != effective; mandatory full-library correlation evidence unavailable otherwise.')
f.to_pickle('scripts/miner_3_20331208_dispersion_conditioned_short_reversal_5d_signal.pkl')
