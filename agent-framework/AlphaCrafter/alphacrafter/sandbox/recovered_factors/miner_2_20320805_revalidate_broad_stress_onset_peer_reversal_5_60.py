"""Revalidation only: broad-stress-onset peer reversal (5,60), using completed data."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
# Persisted construction: relative 5d return observed on onset of broad peer stress,
# accumulated as a 60d event mean and inverted, then one-session lagged.
peer=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A})
q=peer.rolling(60,min_periods=40).quantile(.35)
onset=(peer<q)&~(peer.shift(1)<q.shift(1))
rel5=P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0)
cand=(-rel5.where(onset,axis=0).rolling(60,min_periods=12).mean())
cand=cand.sub(cand.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; y=fw[h].reindex(x.index); v=[]; b=[]
 for d in x.index:
  a=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(a)>=8:
   k=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
   if np.isfinite(k): v.append(k);b.append(len(a))
 v=np.array(v)
 return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR broad_stress_onset_peer_reversal_5_60 REVALIDATION CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
for h in (1,5,10,20):print('H',h,stat(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,p))
mx=-1; who=''; evidence=0
for n,g in S.items():
 a=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic if len(a)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;evidence=len(a)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
