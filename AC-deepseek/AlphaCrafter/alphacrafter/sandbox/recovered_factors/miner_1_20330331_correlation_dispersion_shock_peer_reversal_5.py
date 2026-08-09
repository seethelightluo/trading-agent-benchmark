"""One idea: correlation-dispersion-shock gated peer-relative reversal (5 sessions)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff=z['P'],z['S'],z['A'],z['cutoff']; r=P.pct_change()
# Dispersion of pairwise correlations: a rising value indicates fragmented cross-asset leadership.
disp=pd.Series(index=r.index,dtype=float)
for i in range(19,len(r)):
 c=r.iloc[i-19:i+1].corr().values; disp.iloc[i]=np.std(c[np.triu_indices_from(c,1)])
shock=disp > disp.rolling(60,min_periods=40).quantile(.75).shift(1)
# During fragmentation, favor the recent relative underperformers for a short peer-reversal horizon.
peer5=(P/P.median(axis=1).values.reshape(-1,1)).pct_change(5)
cand=(-peer5.where(shock,np.nan)).sub((-peer5.where(shock,np.nan)).median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];y=fw[h].reindex(x.index);v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   u=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(u):v.append(u);b.append(len(q))
 v=np.asarray(v)
 return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))} if len(v)>1 else {'dates':len(v)}
print('FACTOR correlation_dispersion_shock_peer_reversal_5 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('SHOCK_DATES',int(shock.sum()),'/',len(shock),'RATE',round(float(shock.mean()),6))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME5',n,st(5,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
