"""One idea: moderate VIX-shock post-event peer-relative persistence (60 sessions)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_3_20310821_moderate_yield_spread_postshock_relative_persistence_60.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
# A moderate VIX rise is between its trailing 50th and 85th percentile absolute move.
# After such shocks, score assets by their trailing five-session peer-relative return,
# volatility-scaled, averaged across qualifying episodes; no contemporaneous data used.
vix=z['vix']; vr=vix.pct_change(); av=vr.abs(); q50=av.rolling(60,min_periods=40).quantile(.50);q85=av.rolling(60,min_periods=40).quantile(.85)
event=(vr>0)&(av>=q50)&(av<=q85)
rel5=P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0)
vol20=r.rolling(20,min_periods=15).std()
cand=rel5.div(vol20).where(event.shift(5),axis=0).rolling(60,min_periods=12).mean()
cand=cand.sub(cand.median(axis=1),axis=0).shift(1)
# Include the miner_2 factor admitted after the legacy audit template.
m5=P.pct_change(5).median(axis=1); gate=m5<=m5.rolling(60,min_periods=45).quantile(.20)
S['broad_stress_onset_peer_reversal_5_60']=(-P.pct_change(5).sub(m5,axis=0)).where(gate,axis=0).shift(1)
fw={h:P.shift(-h).div(P).sub(1) for h in (1,5,10,20)}
def stats(h,lo=None,hi=None):
 x=cand.loc[lo:hi] if lo else cand; out=[]; br=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):out.append(v);br.append(len(q))
 if not out:return {'dates':0}
 out=np.asarray(out);return {'dates':len(out),'ic':round(out.mean(),6),'icir':round(out.mean()/out.std(ddof=1),6),'hit':round((out>0).mean(),6),'breadth':round(float(np.mean(br)),3),'min_breadth':int(min(br))}
rk=cand.rank(axis=1,pct=True)
print('FACTOR moderate_vix_shock_postevent_peer_persistence_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('EVENTS',int(event.sum()),'CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(rk.diff().abs().stack().mean()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for nm,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cutoff.date())),('recent180',str(cutoff-pd.Timedelta(days=180)),str(cutoff.date()))]:print('REGIME10',nm,stats(10,lo,hi))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
