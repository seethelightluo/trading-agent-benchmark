"""Revalidation only: admitted inverse cross-asset correlation concentration (40 sessions)."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-05-31')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A});r=p.pct_change()
f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in range(39,len(p)):
 c=r.iloc[t-39:t+1].corr(min_periods=30)
 for a in A:
  z=c.loc[a].drop(a).abs().dropna()
  if len(z)>=8:f.loc[p.index[t],a]=-z.mean()
def calc(h,subset=None):
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for d in f.index if subset is None else f.index[f.index>=subset]:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);sd=x.std()
 return {'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns))}
def regimes(h):
 return {n:calc(h,pd.Timestamp(s)) for n,s in {'2026':'2026-01-01','2027':'2027-01-01','2028_ytd':'2028-01-01','recent_120_sessions':str(p.index[max(0,len(p)-120)].date())}.items()}
turn=[]
for i in range(10,len(f),10):
 z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('REVALIDATION inverse_cross_asset_correlation_concentration_40obs visible',END.date(),'range',p.index.min().date(),p.index.max().date(),'assets',len(A))
print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size),'turnover_10d',float(np.mean(turn)))
for h in [1,5,10,20]: print('METRIC',h,json.dumps(calc(h)),'REGIMES',json.dumps(regimes(h)))
