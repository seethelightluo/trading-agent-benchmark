"""Revalidate one admitted idea: broad stress-onset peer-relative reversal."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
# Shared market panel and operational admitted-library reconstruction.
z=runpy.run_path('scripts/miner_3_20310821_moderate_yield_spread_postshock_relative_persistence_60.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']; m=r.median(axis=1)
ret5=P.pct_change(5); gate=m.rolling(5,min_periods=5).sum() <= m.rolling(5,min_periods=5).sum().rolling(60,min_periods=45).quantile(.20)
# Equivalent to median of 5d asset returns: median(P/P.shift(5)-1), retained directly rather than summed daily median.
broad=ret5.median(axis=1); gate=broad<=broad.rolling(60,min_periods=45).quantile(.20)
cand=(-ret5.sub(broad,axis=0)).where(gate,axis=0).shift(1)
# Amend legacy audit with admitted factors added after its template.
vol20=r.rolling(20,min_periods=15).std(); rel=r.sub(m,axis=0)
S['inverse_peer_relative_lag5_serial_dependence_40']=-(pd.DataFrame({a:rel[a].rolling(40,min_periods=30).corr(rel[a].shift(5)) for a in A}))
S['broad_stress_onset_peer_reversal_5_60']=cand
S['inverse_moderate_vix_shock_postevent_peer_reversal_60']=np.nan # excluded self-like but evidence available as missing is invalid audit, replace calculation
vix=z['vix'];vr=vix.pct_change(); av=vr.abs(); e=(vr>0)&(av>=av.rolling(60,min_periods=40).quantile(.5))&(av<=av.rolling(60,min_periods=40).quantile(.85))
S['inverse_moderate_vix_shock_postevent_peer_reversal_60']=-(P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0).div(vol20).where(e.shift(5),axis=0).rolling(60,min_periods=12).mean()).sub(0) # current separate factor
fw={h:P.shift(-h).div(P).sub(1) for h in (1,5,10,20)}
def st(h,lo=None,hi=None):
 x=cand.loc[lo:hi] if lo else cand; out=[];br=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):out.append(v);br.append(len(q))
 if not out:return {'dates':0}
 x=np.array(out);return {'dates':len(x),'ic':round(x.mean(),6),'icir':round(x.mean()/x.std(ddof=1),6),'hit':round((x>0).mean(),6),'breadth':round(float(np.mean(br)),3),'min_breadth':int(min(br))}
rk=cand.rank(axis=1,pct=True)
print('FACTOR broad_stress_onset_peer_reversal_5_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('GATE_DATES',int(gate.sum()),'/',len(gate),'RATE',round(float(gate.mean()),6))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(rk.diff().abs().stack().mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cutoff.date())),('recent180',str(cutoff-pd.Timedelta(days=180)),str(cutoff.date()))]:print('REGIME10',n,st(10,lo,hi))
mx=-1;who='';ev=0; audited=0
for n,g in S.items():
 if n=='broad_stress_onset_peer_reversal_5_60':continue
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 if len(q)<8:continue
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic;audited+=1
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS_AUDITED',audited)
a=st(10);print('ADMISSION',abs(a['ic'])>=.007 and abs(a['icir'])>=.084 and mx<.5)
