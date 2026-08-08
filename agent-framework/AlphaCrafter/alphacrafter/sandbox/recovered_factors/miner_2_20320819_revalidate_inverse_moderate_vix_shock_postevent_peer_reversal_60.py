"""Revalidation: inverse moderate VIX-shock post-event peer reversal (60 sessions).
Uses the established complete admitted-library reconstruction, plus the previously
admitted broad-stress factor, for the required novelty audit. Only completed
sessions through the simulator cursor are used by the APIs.
"""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_3_20310821_moderate_yield_spread_postshock_relative_persistence_60.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
vix=z['vix']; vr=vix.pct_change(); av=vr.abs(); q50=av.rolling(60,min_periods=40).quantile(.50); q85=av.rolling(60,min_periods=40).quantile(.85)
event=(vr>0)&(av>=q50)&(av<=q85)
rel5=P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0); vol20=r.rolling(20,min_periods=15).std()
cand=(-rel5.div(vol20).where(event.shift(5),axis=0).rolling(60,min_periods=12).mean())
cand=cand.sub(cand.median(axis=1),axis=0).shift(1)
# Add the other admitted miner_2 signal absent from legacy library reconstruction.
m=r.median(axis=1); q20=m.rolling(60,min_periods=40).quantile(.20); stress=m<=q20
S['broad_stress_onset_peer_reversal_5_60']=(-rel5.where(stress).rolling(60,min_periods=12).mean()).sub((-rel5.where(stress).rolling(60,min_periods=12).mean()).median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h).div(P).sub(1) for h in (1,5,10,20)}
def stats(h,lo=None,hi=None):
 x=cand.loc[lo:hi] if lo else cand; out=[]; br=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):out.append(v);br.append(len(q))
 if len(out)<2:return {'dates':len(out)}
 out=np.asarray(out);return {'dates':len(out),'ic':round(float(out.mean()),6),'icir':round(float(out.mean()/out.std(ddof=1)),6),'hit':round(float((out>0).mean()),6),'breadth':round(float(np.mean(br)),3),'min_breadth':int(min(br))}
rk=cand.rank(axis=1,pct=True)
print('FACTOR inverse_moderate_vix_shock_postevent_peer_reversal_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('EVENTS',int(event.sum()),'CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(rk.diff().abs().stack().mean()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for nm,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cutoff.date())),('recent180',str(cutoff-pd.Timedelta(days=180)),str(cutoff.date()))]:print('REGIME10',nm,stats(10,lo,hi))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
s10=stats(10);s20=stats(20)
print('ADMISSION_10',abs(s10['ic'])>=.007 and abs(s10['icir'])>=.084 and mx<.5)
print('ADMISSION_20',abs(s20['ic'])>=.007 and abs(s20['icir'])>=.084 and mx<.5)
