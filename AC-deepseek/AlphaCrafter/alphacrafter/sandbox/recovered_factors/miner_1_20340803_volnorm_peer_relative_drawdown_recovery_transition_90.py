"""One idea: volatility-normalized, peer-relative 10d recovery transition while below a 90d peak."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff,cs=z['P'],z['r'],z['S'],z['A'],z['cutoff'],z['cs']
# A slower recovery transition: recent 10d return improves against prior 10d and peers,
# normalized by own 20d volatility and activated only in a bounded 90d drawdown.
vol=r.rolling(20,min_periods=15).std().replace(0,np.nan)
r10=P/P.shift(10)-1; old10=P.shift(10)/P.shift(20)-1
peer_now=pd.DataFrame({a:r10.drop(columns=a).median(axis=1) for a in A})
peer_old=pd.DataFrame({a:old10.drop(columns=a).median(axis=1) for a in A})
accel=((r10-peer_now)-(old10-peer_old)).div(vol)
dd=(P/P.rolling(90,min_periods=65).max()-1).clip(upper=0)
# Smooth bounded activation emphasizes recoveries from non-trivial but non-crisis losses.
activation=(-dd).clip(lower=0.025,upper=0.25)
cand=cs(accel*activation).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); vals=[]; breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); breadth.append(len(q))
 if not vals:return {'dates':0}
 vals=np.asarray(vals); sd=vals.std(ddof=1)
 return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/sd),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR volnorm_peer_relative_drawdown_recovery_transition_90 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
