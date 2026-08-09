"""One candidate: inverse peer-relative downside/upside return asymmetry (60 sessions).
For each asset the signal is the negative ratio of its trailing semivariance after
negative daily returns to its semivariance after positive returns.  It tests whether
assets with comparatively balanced (rather than downside-dominated) realized paths
subsequently outperform, distinct from event timing and close-location features.
"""
import runpy,numpy as np,pandas as pd,contextlib,io
from scipy.stats import spearmanr
with contextlib.redirect_stdout(io.StringIO()):
 z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
neg=r.where(r<0).pow(2).rolling(60,min_periods=40).mean()
pos=r.where(r>0).pow(2).rolling(60,min_periods=40).mean()
raw=-(neg/(pos+1e-12)).replace([np.inf,-np.inf],np.nan)
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   u=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(u):v.append(u);b.append(len(q))
 v=np.asarray(v)
 return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR inverse_peer_relative_downside_upside_semivariance_ratio_60 CUTOFF',cutoff.date(),'ASSETS',len(A),'PERIOD',P.index.min().date(),cutoff.date())
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20):print('H',h,stat(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
