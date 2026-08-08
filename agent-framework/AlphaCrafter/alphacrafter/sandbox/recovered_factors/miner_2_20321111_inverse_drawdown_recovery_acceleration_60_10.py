"""One idea: inverse drawdown-recovery acceleration conditional on prior drawdown burden.
Assets whose 60-session drawdown has improved quickly relative to their own prior
underwater duration may exhibit an overextended recovery and underperform peers.
This differs from a level/duration measure by using the *change* in drawdown.
"""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
peak=P.rolling(60,min_periods=45).max(); dd=P/peak-1
# Completed-session recovery speed; normalize by prior drawdown depth and require
# a meaningful prior underwater state, while retaining a neutral missing signal.
prior=dd.shift(10)
raw=-(dd-prior)/(0.01-prior)
raw=raw.where(prior < -0.01)
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
# Include the recently admitted duration/depth signal in novelty comparison.
below=P.lt(peak)
dur=pd.DataFrame({a:below[a].astype(int).groupby((~below[a]).cumsum()).cumsum() for a in A})
S['inverse_drawdown_duration_burden_depth_adjusted_60']=(dur/(1+100*(-dd))).sub((dur/(1+100*(-dd))).median(axis=1),axis=0).shift(1)
fw={h:pd.DataFrame({a:(P[a].dropna().shift(-h)/P[a].dropna()-1).reindex(P.index) for a in A}) for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; y=fw[h].reindex(x.index); vals=[];breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vals.append(v);breadth.append(len(q))
 vals=np.asarray(vals)
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))} if len(vals)>1 else {'dates':len(vals)}
print('FACTOR inverse_drawdown_recovery_acceleration_60_10 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
