"""One idea: peer-residual short-horizon reversal (5/40 observations).
At each date, rank assets by the negative of their last five-session return after
removing the return implied by a trailing 40-session beta to the equal-weight
cross-asset peer benchmark.  This seeks idiosyncratic overshoots rather than
absolute reversal or volatility-normalized reversal.
"""
import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-01-24')
def close(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date)
 return pd.to_numeric(d.set_index('date').loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:close(a) for a in A}).sort_index(); r=p.pct_change()
# Equal-weight peer (own asset excluded) avoids a self-referential benchmark.
f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for k in range(45,len(p)):
 for a in A:
  peer=r.iloc[k-40:k].drop(columns=a).mean(axis=1)
  own=r[a].iloc[k-40:k]; z=pd.concat([own.rename('x'),peer.rename('m')],axis=1).dropna()
  if len(z)<30: continue
  v=z.m.var()
  if not np.isfinite(v) or v==0: continue
  beta=z.x.cov(z.m)/v
  recent_own=r[a].iloc[k-5:k].sum(); recent_peer=r.drop(columns=a).iloc[k-5:k].mean(axis=1).sum()
  f.loc[p.index[k],a]=-(recent_own-beta*recent_peer)
def metric(h):
 fw=p.shift(-h).div(p).sub(1); out=[]; ns=[]
 for d in p.index:
  z=pd.concat([f.loc[d].rename('factor'),fw.loc[d].rename('forward')],axis=1).dropna()
  if len(z)>=8:
   q=z.factor.corr(z.forward,method='spearman')
   if np.isfinite(q): out.append((d,q)); ns.append(len(z))
 x=pd.Series(dict(out),dtype=float); sd=x.std()
 regs={}
 for name, mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('2029_ytd',x.index.year==2029),('latest120',np.arange(len(x))>=max(0,len(x)-120))]:
  q=x[mask]; regs[name]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if len(q)>1 else np.nan,'hit_ratio':(q>0).mean() if len(q) else np.nan}
 turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turns)),'regimes':regs}
print('FACTOR peer_residual_short_horizon_reversal_5_40obs')
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'valid_cells',int(f.count().sum()),'of',f.size,'coverage',float(f.count().sum()/f.size))
for h in (1,5,10,20): print('METRIC',json.dumps(metric(h),default=float))
# Non-admission related-signal diagnostics.
rev=-p.pct_change(5); mom=p.pct_change(20)
for n,g in [('raw_reversal_5obs',rev),('momentum_20obs',mom)]:
 z=pd.concat([f.stack().rename('candidate'),g.stack().rename(n)],axis=1).dropna()
 print('SCREEN_CORR',n,'rho',z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),'cells',len(z))
