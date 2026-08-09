"""One candidate: residual trend-efficiency (20d), completed bars through validation cutoff."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-12-21')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change(); resid=r.sub(r.mean(axis=1),axis=0)
# Signed net residual movement divided by total residual movement: idiosyncratic trend persistence, scale invariant.
f=resid.rolling(20,min_periods=20).sum().div(resid.abs().rolling(20,min_periods=20).sum()).replace([np.inf,-np.inf],np.nan)
print('FACTOR residual_trend_efficiency_20d VALIDATED_THROUGH',CUT.date())
print('definition=sum(residual returns,20)/sum(abs(residual returns),20); residual = asset daily return minus equal-weight cross-asset daily return')
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
for label,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01',CUT)]:
 s=stats[10].loc[lo:hi]
 print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(label,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turns),len(turns)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn))
  if x.get('validation',{}).get('status')=='EFFECTIVE':eff.append(x['factor_id'])
 except Exception: pass
found=[]; scores=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if hits:
  found.append(fid);z=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1:scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE artifacts=%d effective=%d'%(len(found),len(eff)))
if scores: print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
print('ADMISSION=FAIL if artifacts != effective; complete maximum correlation evidence is mandatory.')
f.to_pickle('scripts/miner_2_20331222_residual_trend_efficiency_20d_signal.pkl')
