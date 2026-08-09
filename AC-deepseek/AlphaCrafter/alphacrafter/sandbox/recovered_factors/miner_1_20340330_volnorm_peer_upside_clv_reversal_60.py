"""One idea: volatility-normalized continuous peer-upside close-location reversal (60d).
This is deliberately the signed reversal of the previously rejected upside-continuation
hypothesis; it is evaluated as a separately named factor."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_1_20340316_volnorm_peer_upside_clv_60.py')
# The sourced candidate is the fully residualized, implementation-lagged continuation score.
# Negation defines the separate reversal hypothesis without changing inputs or timing.
cand=-z['cand']; P=z['P']; S=z['S']; A=z['A']; cutoff=z['cutoff']
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; y=fw[h].reindex(x.index)
 vals=[]; breadth=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(rho): vals.append(rho); breadth.append(len(q))
 if not vals:return {'dates':0}
 vals=np.asarray(vals)
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('\nREVERSAL_FACTOR volnorm_continuous_peer_upside_close_location_reversal_60 VALIDATION_CUTOFF',cutoff.date(),'ASSETS',len(A),'LIBRARY',len(S))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
for h in (1,5,10,20): print('H',h,stat(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,stat(10,p))
mx=-1; who=''; evidence=0
for name,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 if len(q)<8: continue
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 if np.isfinite(rho) and abs(rho)>mx: mx,who,evidence=abs(rho),name,len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',evidence)
