"""One idea: fixed-window broad-downturn relative-capture reversal, 10d.
During a completed five-session broad downturn, assets with better ten-session
peer-relative performance may retain defensive sponsorship and mean revert less."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
# Fixed, observable broad weakness state; no intensity weighting or long-state filter.
broad5=r.median(axis=1).rolling(5,min_periods=4).sum()
state=broad5 < -0.012
rel10=P.pct_change(10).sub(P.pct_change(10).median(axis=1),axis=0)
cand=rel10.where(state,np.nan).shift(1)
fw={h:P.shift(-h).div(P).sub(1) for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); vals=[]; breadth=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); breadth.append(len(q))
 vals=np.asarray(vals)
 if not len(vals): return {'dates':0}
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR fixed_window_broad_downturn_relative_capture_10 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('STATE_DATES',int(state.sum()),'STATE_RATE',round(float(state.mean()),6))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20): print('H',h,stats(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,stats(10,p))
mx=-1; who=''; evidence=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q) else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx: mx=abs(rho); who=n; evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
