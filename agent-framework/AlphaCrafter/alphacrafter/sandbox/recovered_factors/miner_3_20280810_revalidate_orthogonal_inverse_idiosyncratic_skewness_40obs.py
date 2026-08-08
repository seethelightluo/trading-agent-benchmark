"""Revalidation/admission: orthogonal inverse idiosyncratic skewness 40d.
Uses data through 2028-08-09 and tests 1/5/10/20-day forward rank ICs.  The raw
signal is negative skew of a 40d residual-to-peer return series. Each date it is
residualized cross-sectionally against expected-shortfall, downside peer
correlation, inverse excess kurtosis and risk-adjusted trend."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-08-09')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index); return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A}); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(40 if c is not None else 20,min_periods=12 if c is not None else 15).cov(y)/y.rolling(40 if c is not None else 20,min_periods=12 if c is not None else 15).var()
idres=pd.DataFrame({a:r[a]-beta(r[a],peer[a])*peer[a] for a in A}); raw=-idres.rolling(40,min_periods=30).skew()
trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
downp=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A}); kurt=-r.rolling(40,min_periods=30).kurt()
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),es.loc[d].rename('es'),downp.loc[d].rename('dp'),kurt.loc[d].rename('k'),trend.loc[d].rename('tr')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['es','dp','k','tr']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def metrics(h):
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(v):out.append((d,v));ns.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(); regs={}
 for name, yy in [('2026',[2026]),('2027',[2027]),('2028_ytd',[2028]),('latest_120',None)]:
  q=x.tail(120) if yy is None else x[x.index.year.isin(yy)]
  regs[name]={'dates':len(q),'ic':q.mean() if len(q) else None,'icir':q.mean()/q.std() if len(q)>1 and q.std()>0 else None}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'ic_se':sd/np.sqrt(len(x)),'mean_instruments':np.mean(ns),'turnover_10d':np.mean(turn),'regimes':regs}
print('CANDIDATE visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date()); print('COVERAGE',int(f.count().sum()),'/',f.size,f.count().sum()/f.size)
for h in [1,5,10,20]: print('METRIC',json.dumps(metrics(h),default=float))
# exact currently active signal forms; deprecated relative-volume and correlation-concentration excluded
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8: orth.loc[d,z.index]=z.iloc[:,0]-np.c_[np.ones(len(z)),z.iloc[:,1]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1]],z.iloc[:,0],rcond=None)[0]
di=get_index_daily_data('DXY',5000).set_index('date');di.index=pd.to_datetime(di.index); dx=pd.to_numeric(di.loc[:END,'close'],errors='coerce').pct_change().reindex(r.index)
spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A}); aut=r.rolling(20,min_periods=15).corr(r.shift(1)); up=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A}); asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A}); du=pd.DataFrame({a:-beta(r[a],dx,dx>0) for a in A}); dd=pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A})
q=r.quantile(.2,axis=1); per=r.le(q,axis=0).astype(float).rolling(60,min_periods=40).mean(); pt=per*np.nan
for d in p.index:
 z=pd.concat([per.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8: pt.loc[d,z.index]=z.iloc[:,0]-np.c_[np.ones(len(z)),z.iloc[:,1]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1]],z.iloc[:,0],rcond=None)[0]
# active library excluding duplicated ravmom (identical signal) is nevertheless tested separately
lib={'ravmom':trend,'risk_adjusted_trend':trend,'volnorm_reversal':rev,'downside_peer_correlation':downp,'return_autorrelation':aut,'orthogonal_acceleration':orth,'negative_spx_beta':spx,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'inverse_upside_peer_correlation':up,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'asymmetric_peer_beta_resilience':asym,'peer_downside_tail_persistence_residual':pt}
mx=-1; who=None
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); rho=z.f.corr(z.x,method='spearman'); print('LIB',n,rho,len(z))
 if not np.isfinite(rho): raise RuntimeError('missing correlation evidence '+n)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
