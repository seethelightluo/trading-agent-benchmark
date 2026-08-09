"""One idea: CN10Y-relative trend dispersion.
Score is each asset's 20-session return less CN10Y's simultaneous 20-session return.
This is a continuous, non-beta CN10Y transmission-relative trend measure."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-12-27')
def close(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date)
 return pd.to_numeric(d.set_index('date').loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:close(a) for a in A}); r=p.pct_change()
f=p.pct_change(20).sub(p.CN10Y.pct_change(20),axis=0)
def metric(h):
 fw=p.shift(-h).div(p).sub(1); out=[]; ns=[]
 for d in p.index:
  z=pd.concat([f.loc[d].rename('factor'),fw.loc[d].rename('forward')],axis=1).dropna()
  if len(z)>=8:
   q=z.factor.corr(z.forward,method='spearman')
   if np.isfinite(q):out.append((d,q));ns.append(len(z))
 x=pd.Series(dict(out)); sd=x.std()
 reg={}
 for name,mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('latest120',np.arange(len(x))>=len(x)-120)]:
  q=x[mask];reg[name]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if len(q)>1 else np.nan,'hit_ratio':(q>0).mean() if len(q) else np.nan}
 turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_instruments':np.mean(ns),'turnover_10d':np.mean(turns),'regimes':reg}
print('FACTOR cn10y_relative_trend_dispersion_20obs')
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'valid_cells',int(f.count().sum()),'of',f.size,'coverage',f.count().sum()/f.size)
for h in (1,5,10,20): print('METRIC',json.dumps(metric(h),default=float))
# Key related-signal diagnostic (not an admission substitute): conventional 20d trend.
mom=p.pct_change(20)
z=pd.concat([f.stack().rename('candidate'),mom.stack().rename('momentum20')],axis=1).dropna()
print('SCREEN_CORR momentum_20obs rho',z.candidate.corr(z.momentum20,method='spearman'),'cells',len(z))
