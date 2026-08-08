"""One factor: continuous peer-downside capture acceleration (20 vs 60 sessions)."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
# On each day weight an asset's peer-relative return by the continuous magnitude
# of broad peer weakness. Recent minus long-run capture measures improving defense.
peer_sum=r.sum(axis=1, min_count=2)
peer_n=r.notna().sum(axis=1)-1
peer_mean=r.rsub(peer_sum, axis=0).div(peer_n, axis=0)
rel=r-peer_mean
broad=r.median(axis=1)
# Scale downside intensity by its trailing typical absolute market move, preventing
# a few crisis days from entirely determining the estimate.
scale=broad.abs().rolling(60,min_periods=30).median().replace(0,np.nan)
w=(-broad.clip(upper=0)/scale).clip(upper=4).fillna(0)
def wavg(x,win):
 num=x.mul(w,axis=0).rolling(win,min_periods=max(12,win//2)).sum()
 den=w.rolling(win,min_periods=max(12,win//2)).sum().replace(0,np.nan)
 return num.div(den,axis=0)
raw=wavg(rel,20)-wavg(rel,60)
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1];] if False else (cand if period is None else cand.loc[period[0]:period[1]])
 y=fw[h].reindex(x.index); vals=[]; breadth=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); breadth.append(len(q))
 vals=np.array(vals)
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR continuous_peer_downside_capture_acceleration_20_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20): print('H',h,st(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q) else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx: mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
