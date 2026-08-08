"""One idea: CN10Y shock-response asymmetry (40 observations).
For each asset, score its average return on CN10Y large-up-shock days minus
its average return on CN10Y large-down-shock days in the trailing 40 sessions.
This measures relative cross-asset transmission asymmetry, rather than trend.
"""
import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-01-10')
def getclose(a):
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date)
 return pd.to_numeric(d.set_index('date').loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:getclose(a) for a in A}); r=p.pct_change(); y=r['CN10Y']
# At date t, use only return observations t-40 ... t-1. Shock definitions are
# cross-time empirical quartiles within this same trailing window.
f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for k in range(41,len(p)):
 hist=y.iloc[k-40:k].dropna()
 if len(hist)<30: continue
 lo,hi=hist.quantile(.25),hist.quantile(.75)
 up=r.iloc[k-40:k].loc[hist.index][hist>=hi].mean()
 dn=r.iloc[k-40:k].loc[hist.index][hist<=lo].mean()
 f.iloc[k]=up-dn
def metric(h):
 fw=p.shift(-h).div(p).sub(1); vals=[];ns=[]
 for d in p.index:
  z=pd.concat([f.loc[d].rename('factor'),fw.loc[d].rename('forward')],axis=1).dropna()
  if len(z)>=8:
   q=z.factor.corr(z.forward,method='spearman')
   if np.isfinite(q): vals.append((d,q));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std()
 regimes={}
 for name,mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('latest120',np.arange(len(x))>=len(x)-120)]:
  q=x[mask]; regimes[name]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if len(q)>1 else np.nan,'hit_ratio':(q>0).mean() if len(q) else np.nan}
 turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turns)),'regimes':regimes}
print('FACTOR cn10y_shock_response_asymmetry_40obs')
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'valid_cells',int(f.count().sum()),'of',f.size,'coverage',float(f.count().sum()/f.size))
for h in (1,5,10,20): print('METRIC',json.dumps(metric(h),default=float))
# Related but non-admission diagnostic: trailing simple momentum correlation.
mom=p.pct_change(20);z=pd.concat([f.stack().rename('candidate'),mom.stack().rename('mom20')],axis=1).dropna()
print('SCREEN_CORR momentum_20obs rho',z.candidate.corr(z.mom20,method='spearman'),'cells',len(z))
