"""One idea: peer-market tail beta asymmetry, 60 sessions.
Measures each asset's relative participation in broad-peer downside versus upside tails.
"""
import runpy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
# Leave-one-out peer median prevents the asset's own return entering its market-state definition.
raw=pd.DataFrame(index=r.index,columns=A,dtype=float)
for a in A:
    peers=r.drop(columns=a).median(axis=1)
    lo=peers.rolling(60,min_periods=40).quantile(.20).shift(1)
    hi=peers.rolling(60,min_periods=40).quantile(.80).shift(1)
    down=r[a].where(peers<=lo).rolling(60,min_periods=12).mean()
    up=r[a].where(peers>=hi).rolling(60,min_periods=12).mean()
    # Difference tests whether relative tail participation asymmetry predicts future ranking.
    raw[a]=down-up
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); v=[];breadth=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   k=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(k):v.append(k);breadth.append(len(q))
 v=np.asarray(v)
 return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))} if len(v)>1 else {'dates':len(v)}
print('FACTOR peer_market_tail_beta_asymmetry_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20): print('H',h,stats(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,stats(10,p))
mx=-1.;who='';ev=0;invalid=[]
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if not np.isfinite(rho): invalid.append(n)
 elif abs(rho)>mx: mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S),'INVALID',invalid)
