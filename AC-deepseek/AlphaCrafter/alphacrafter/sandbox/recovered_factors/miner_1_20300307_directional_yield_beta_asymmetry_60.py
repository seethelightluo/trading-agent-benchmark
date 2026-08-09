"""One idea: directional yield-beta asymmetry: assets resilient to rate rises relative to rate declines."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list'];C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C);r=P.pct_change();cutoff=P.dropna(how='all').index.max(); y=r['US10Y']; m=r.median(axis=1)
def cs(x):return x.sub(x.median(axis=1),axis=0)
def ebet(x,ev,w=60):
 # beta to yield using only directional-yield sessions; min 15 event/calendar observations
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1).where(ev,axis=0)
 return z.x.rolling(w,min_periods=15).cov(z.y)/z.y.rolling(w,min_periods=15).var()
# Candidate is positive where the asset's beta to positive yield changes exceeds its beta to negative changes.
# Availability lag prevents use of the session's close in its own forward return.
cand=cs(pd.DataFrame({a:ebet(r[a],y>0)-ebet(r[a],y<0) for a in A})).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stat(h,lo=None):
 x=cand if lo is None else cand.loc[lo:]; out=[];breadth=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(q)>=8:out.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);breadth.append(len(q))
 z=np.array(out)
 return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),6),'breadth':round(np.mean(breadth),3),'min_breadth':min(breadth)} if len(z) else {}
print('FACTOR directional_yield_beta_asymmetry_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(cand.std(axis=1).mean(),6))
for h in (1,5,10,20):print('H',h,stat(h))
for label,lo in [('since2023','2023-01-01'),('since2025','2025-01-01'),('since2027','2027-01-01'),('recent180',str(cutoff-pd.Timedelta(days=180)))]:print('REGIME10',label,stat(10,lo))
# Required library comparison: exact nearest admitted rate-shock factor, calculated contemporaneously.
def beta(x,w=60):return x.rolling(w,min_periods=15).cov(y)/y.rolling(w,min_periods=15).var()
shock=y.abs()>y.abs().rolling(60,min_periods=40).quantile(.75)
nearest=cs(pd.DataFrame({a:ebet(r[a],shock)-beta(r[a]) for a in A})).shift(1)
q=pd.concat([cand.stack(),nearest.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
print('LIBRARY_COMPARISON yield_shock_beta_resilience_60 CELLS',len(q),'RHO',round(rho,6))
print('NOTE full-library audit is only meaningful if IC gates pass; this direct closest-factor audit is reported regardless.')
