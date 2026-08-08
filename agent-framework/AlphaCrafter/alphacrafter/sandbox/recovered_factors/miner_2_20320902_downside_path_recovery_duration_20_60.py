"""One idea: downside-path recovery duration, 60 sessions.
For each asset, identify completed sessions that remain below their trailing
20-session high after a negative five-session return.  The signal is the
negative mean number of consecutive below-high sessions (capped at 20) over
60 sessions, then cross-sectionally median centred.  Higher values mean a
shorter, more resilient recovery path after declines.  All ingredients are
lagged one completed session before forward-return testing.
"""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
x={}
for a in A:
    p=P[a]; peak=p.rolling(20,min_periods=15).max()
    below=(p<peak).astype(float)
    # Consecutive completed days below the rolling high; reset at a new high.
    grp=(below==0).cumsum()
    duration=below.groupby(grp).cumsum().clip(upper=20)
    drawdown_episode=(p/p.shift(5)-1<0) & (below>0)
    x[a]=(-duration.where(drawdown_episode)).rolling(60,min_periods=12).mean()
cand=pd.DataFrame(x).sub(pd.DataFrame(x).median(axis=1),axis=0).shift(1)
fw={k:P.shift(-k)/P-1 for k in (1,5,10,20)}
def stats(k,period=None):
 qx=cand if period is None else cand.loc[period[0]:period[1]]; qy=fw[k].reindex(qx.index); vals=[]; breadth=[]
 for dt in qx.index:
  q=pd.concat([qx.loc[dt],qy.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   vv=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(vv): vals.append(vv); breadth.append(len(q))
 vals=np.asarray(vals)
 if len(vals)<2:return {'dates':len(vals)}
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR downside_path_recovery_duration_20_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for k in (1,5,10,20): print('H',k,stats(k))
for n,pd_ in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,stats(10,pd_))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
