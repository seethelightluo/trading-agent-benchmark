"""One factor: peer-downside capture acceleration, 20 versus 60 sessions."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
# On completed sessions where the other 14 assets are in their trailing lower quintile,
# measure each asset's relative return. Improvement in 20d capture over 60d capture
# identifies recent defensive/resilient acceleration rather than absolute trend.
peer=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A})
rel=r-peer
flag=pd.DataFrame({a:peer[a]<peer[a].rolling(60,min_periods=40).quantile(.20) for a in A})
def emean(w,minimum): return pd.DataFrame({a:rel[a].where(flag[a].shift(1)).rolling(w,min_periods=minimum).mean() for a in A})
raw=emean(20,5)-emean(60,15)
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]
 y=fw[h].reindex(x.index); ic=[]; br=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): ic.append(v);br.append(len(q))
 ic=np.array(ic)
 return {'dates':len(ic),'ic':round(float(ic.mean()),6),'icir':round(float(ic.mean()/ic.std(ddof=1)),6),'hit':round(float((ic>0).mean()),6),'mean_breadth':round(float(np.mean(br)),3),'min_breadth':int(min(br))} if len(ic) else {'dates':0}
print('FACTOR peer_downside_capture_acceleration_20_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20): print('H',h,st(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q) else np.nan
 if np.isfinite(rho) and abs(rho)>mx: mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
