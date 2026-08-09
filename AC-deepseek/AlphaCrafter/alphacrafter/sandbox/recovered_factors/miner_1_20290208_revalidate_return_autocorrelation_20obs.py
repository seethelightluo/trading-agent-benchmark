"""Revalidation of one admitted idea: 20-session return autocorrelation."""
import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-02-07')
def rd(a):
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date)
 return pd.to_numeric(d.set_index('date').loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A}).sort_index();r=p.pct_change()
# Value at t uses only return pairs ending at t; higher serial persistence is the validated direction.
f=r.rolling(20,min_periods=16).corr(r.shift(1))
def met(h):
 fw=p.shift(-h).div(p).sub(1); obs=[]; ns=[]
 for d in p.index:
  z=pd.concat([f.loc[d].rename('factor'),fw.loc[d].rename('forward')],axis=1).dropna()
  if len(z)>=8:
   ic=z.factor.corr(z.forward,method='spearman')
   if np.isfinite(ic):obs.append((d,ic));ns.append(len(z))
 x=pd.Series(dict(obs)); sd=x.std(); regs={}
 for name,mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('2029_ytd',x.index.year==2029),('latest120',np.arange(len(x))>=max(0,len(x)-120))]:
  y=x[mask];regs[name]={'dates':len(y),'ic':y.mean(),'icir':y.mean()/y.std() if len(y)>1 and y.std()>0 else np.nan,'hit_ratio':(y>0).mean() if len(y) else np.nan}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':regs}
print('FACTOR return_autocorrelation_20obs REVALIDATION')
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'valid_cells',int(f.count().sum()),'of',f.size,'coverage',float(f.count().sum()/f.size))
for h in (1,5,10,20):print('METRIC',json.dumps(met(h),default=float))
# Required library-overlap evidence: reconstruction of factor's nearest same-miner admitted signals.
vol=r.rolling(20,min_periods=15).std();trend=(p/p.shift(20)-1)/vol
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
for name,g in {'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':down}.items():
 z=pd.concat([f.stack().rename('f'),g.stack().rename('g')],axis=1).dropna()
 print('LIB_CORR',name,'rho',z.f.corr(z.g,method='spearman'),'cells',len(z))
