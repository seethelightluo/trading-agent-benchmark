"""One idea: peer-relative next-session rebound after an asset-specific downside shock.
Signal is each asset's mean peer-relative return immediately following its own
bottom-quintile return events in the preceding 60 sessions. It tests whether
post-shock recovery propensity predicts subsequent cross-asset returns."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
rel=r.sub(r.median(axis=1),axis=0)
f={}
for a in A:
    threshold=r[a].rolling(60,min_periods=40).quantile(.20).shift(1)
    # At t, include return t only if t-1 was a completed asset-specific shock.
    f[a]=rel[a].where(r[a].shift(1)<threshold.shift(1)).rolling(60,min_periods=12).mean()
cand=pd.DataFrame(f).sub(pd.DataFrame(f).median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; y=fw[h].reindex(x.index); vals=[];breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);breadth.append(len(q))
 vals=np.array(vals)
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))} if len(vals)>1 else {'dates':len(vals)}
print('FACTOR post_own_downside_shock_peer_relative_rebound_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20): print('H',h,stat(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,stat(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx: mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
