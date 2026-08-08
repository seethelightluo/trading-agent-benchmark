"""Explore inverse downside-tail concentration, 60 sessions. Completed-session signals only."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff=z['P'],z['S'],z['A'],z['cutoff']
# A low value means adverse squared-return risk is concentrated in a few crash days;
# negate it so dispersed/milder downside paths rank positively.
r=P.pct_change()
f=pd.DataFrame(np.nan,index=P.index,columns=A)
for a in A:
 neg=(-r[a].clip(upper=0)).pow(2)
 total=neg.rolling(60,min_periods=45).sum()
 worst=neg.rolling(60,min_periods=45).apply(lambda x: np.sort(x)[-5:].sum(),raw=True)
 f[a]=-(worst/(total+1e-12))
cand=f.sub(f.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); v=[]; b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   w=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(w):v.append(w);b.append(len(q))
 v=np.array(v)
 return dict(dates=len(v),ic=round(float(v.mean()),6),icir=round(float(v.mean()/v.std(ddof=1)),6),hit=round(float((v>0).mean()),6),mean_breadth=round(float(np.mean(b)),3),min_breadth=int(min(b)))
print('FACTOR inverse_downside_tail_concentration_5of60 VALIDATION_DATE 2033-02-03 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME20',n,stats(20,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
