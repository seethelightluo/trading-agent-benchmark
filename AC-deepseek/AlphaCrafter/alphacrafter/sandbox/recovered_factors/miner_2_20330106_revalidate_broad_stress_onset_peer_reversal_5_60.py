"""Revalidation only: broad-stress-onset peer-relative reversal, current visible cutoff."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_3_20310821_moderate_yield_spread_postshock_relative_persistence_60.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
ret5=P.pct_change(5); broad=ret5.median(axis=1); threshold=broad.rolling(60,min_periods=45).quantile(.20); gate=broad<=threshold
cand=(-ret5.sub(ret5.median(axis=1),axis=0)).where(gate,0.0).shift(1)
# Include the other Miner 2 VIX event signal absent from the legacy reconstructed library.
vix=z['vix']; vr=vix.pct_change(); av=vr.abs(); event=(vr>0)&(av>=av.rolling(60,min_periods=40).quantile(.5))&(av<=av.rolling(60,min_periods=40).quantile(.85))
rel5=ret5.sub(ret5.median(axis=1),axis=0); vol20=r.rolling(20,min_periods=15).std()
S['inverse_moderate_vix_shock_postevent_peer_reversal_60']=(-rel5.div(vol20).where(event.shift(5),axis=0).rolling(60,min_periods=12).mean()).sub((-rel5.div(vol20).where(event.shift(5),axis=0).rolling(60,min_periods=12).mean()).median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h).div(P).sub(1) for h in (1,5,10,20)}
def stat(h,lo=None,hi=None):
 x=cand.loc[lo:hi] if lo else cand; v=[]; b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   a=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(a):v.append(a);b.append(len(q))
 if len(v)<2:return {'dates':len(v)}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
rk=cand.rank(axis=1,pct=True)
print('FACTOR broad_stress_onset_peer_reversal_5_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('GATE_DATES',int(gate.sum()),'/',len(gate),'RATE',round(float(gate.mean()),6),'CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(rk.diff().abs().stack().mean()),6))
for h in (1,5,10,20):print('H',h,stat(h))
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cutoff.date())),('recent180',str(cutoff-pd.Timedelta(days=180)),str(cutoff.date()))]:print('REGIME20',n,stat(20,lo,hi))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
for h in (10,20):
 s=stat(h);print('ADMISSION',h,abs(s['ic'])>=.007 and abs(s['icir'])>=.084 and mx<.5)
