"""One idea: inverse cross-asset correlation-centrality expansion (20 vs 60 sessions).
An asset whose return correlation to the other fourteen assets has risen sharply
may be losing diversification and become relatively vulnerable.  The negative
change in mean pairwise correlation is an interpretable cross-asset topology
signal, lagged one full session for a completed-data decision."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff=z['P'],z['S'],z['A'],z['cutoff']; r=P.pct_change()
def centrality(w):
 out=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in range(w,len(r)+1):
  c=r.iloc[t-w:t].corr()
  out.iloc[t-1]=[(c.loc[a].drop(a).mean()) for a in A]
 return out
# The difference only uses trailing returns ending at date t; shift ensures t's
# close itself is unavailable at the next decision.
short=centrality(20); long=centrality(60)
cand=-(short-long).sub((short-long).median(axis=1),axis=0).shift(1)
fw={h:P.pct_change(h).shift(-h) for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];y=fw[h].reindex(x.index);v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   k=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(k):v.append(k);b.append(len(q))
 if len(v)<2:return {'dates':len(v)}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR inverse_correlation_centrality_expansion_20_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stats(10,p))
mx=-1;who='';evidence=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
