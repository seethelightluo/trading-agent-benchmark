"""One idea: USDCNY directional-return asymmetry (60 sessions), with full library novelty audit."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data
# Reuse the current complete operational admitted-factor reconstruction, ensuring identical visible data and audit set.
z=runpy.run_path('scripts/miner_3_20310821_moderate_yield_spread_postshock_relative_persistence_60.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
d=get_index_daily_data('USDCNY',5000).copy();d.date=pd.to_datetime(d.date)
fx=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce').reindex(P.index); fr=fx.pct_change()
# Higher score: comparatively better mean daily returns when CNY strengthens (USDCNY falls) than when it weakens. Each side needs 15 observations.
cand=pd.DataFrame({a:r[a].where(fr<0).rolling(60,min_periods=15).mean()-r[a].where(fr>0).rolling(60,min_periods=15).mean() for a in A})
cand=cand.sub(cand.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h).div(P).sub(1) for h in (1,5,10,20)}
def st(h,lo=None,hi=None):
 x=cand.loc[lo:hi] if lo else cand; out=[];breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):out.append(v);breadth.append(len(q))
 out=np.array(out)
 return {'dates':len(out),'ic':round(out.mean(),6),'icir':round(out.mean()/out.std(ddof=1),6),'hit':round((out>0).mean(),6),'breadth':round(np.mean(breadth),3),'min_breadth':min(breadth)}
rank=cand.rank(axis=1,pct=True)
print('FACTOR usdcny_directional_return_asymmetry_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(rank.diff().abs().stack().mean(),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cutoff.date())),('recent180',str(cutoff-pd.Timedelta(days=180)),str(cutoff.date()))]:print('REGIME10',n,st(10,lo,hi))
mx=0;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q) else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
