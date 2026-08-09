"""One idea: unconditional CN10Y transmission resilience, 60 observations.
Higher score denotes a lower rolling sensitivity of asset returns to CN10Y changes."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-11-29')
def close(a):
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date)
 return pd.to_numeric(d.set_index('date').loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:close(a) for a in A});r=p.pct_change(); ix=p.index;cn=r.CN10Y
f=pd.DataFrame({a:-r[a].rolling(60,min_periods=30).cov(cn)/cn.rolling(60,min_periods=30).var().replace(0,np.nan) for a in A})
def met(h):
 fw=p.shift(-h)/p-1; vals=[];ns=[]
 for d in ix:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):vals.append((d,q));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(); turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):turns.append(1-q)
 regs={}
 for n,m in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('latest120',np.arange(len(x))>=len(x)-120)]:
  q=x[m];regs[n]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if len(q)>1 else np.nan,'hit':(q>0).mean() if len(q) else np.nan}
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_instruments':np.mean(ns),'turnover_10d':np.mean(turns),'regimes':regs}
print('FACTOR cn10y_unconditional_transmission_resilience_60obs');print('VISIBLE',END.date(),'assets',len(A),'dates',len(p),'cells',int(f.count().sum()),'of',f.size,'coverage',f.count().sum()/f.size)
for h in (1,5,10,20):print('METRIC',json.dumps(met(h),default=float))
