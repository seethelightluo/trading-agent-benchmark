"""One candidate: orthogonal peer-beta compression (20 minus 60 sessions).
A high value denotes an asset whose sensitivity to its cross-asset peer basket has recently
fallen relative to its medium-run sensitivity, residualized against existing risk/trend signals."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-09-20')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
dat={a:rd(a) for a in A}; p=pd.DataFrame({a:pd.to_numeric(dat[a].close,errors='coerce') for a in A});r=p.pct_change();vol=r.rolling(20,min_periods=15).std();peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,w=40,mp=15,cond=None):
 if cond is not None:x=x.where(cond);y=y.where(cond)
 return x.rolling(w,min_periods=mp).cov(y)/y.rolling(w,min_periods=mp).var().replace(0,np.nan)
trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
downp=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});kurt=-r.rolling(40,min_periods=30).kurt()
b20=pd.DataFrame({a:beta(r[a],peer[a],20,15) for a in A});b60=pd.DataFrame({a:beta(r[a],peer[a],60,45) for a in A});raw=b60-b20
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),es.loc[d].rename('es'),downp.loc[d].rename('dp'),kurt.loc[d].rename('k'),trend.loc[d].rename('tr')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['es','dp','k','tr']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def metric(h):
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));sd=x.std();regs={}
 for n,sel in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('latest_120',np.arange(len(x))>=len(x)-120)]:
  q=x[sel];regs[n]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if len(q)>1 else np.nan}
 turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'ic_se':sd/np.sqrt(len(x)),'mean_instruments':np.mean(ns),'turnover_10d':np.mean(turns),'regimes':regs}
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'candidate_cells',int(f.count().sum()),'of',f.size)
for h in [1,5,10,20]:print('METRIC',json.dumps(metric(h),default=float))
# reconstruct admitted library signals for mandatory full correlation evidence
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:orth.loc[d,z.index]=z.iloc[:,0]-np.c_[np.ones(len(z)),z.iloc[:,1]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1]],z.iloc[:,0],rcond=None)[0]
di=get_index_daily_data('DXY',5000).set_index('date');di.index=pd.to_datetime(di.index);dx=pd.to_numeric(di.loc[:END,'close'],errors='coerce').pct_change().reindex(r.index)
spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],40,12,peer[a]<0)-beta(r[a],peer[a],40,12,peer[a]>0) for a in A});du=pd.DataFrame({a:-beta(r[a],dx,40,12,dx>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dx,40,12,dx<0) for a in A})
q=r.quantile(.20,axis=1);persist=r.le(q,axis=0).astype(float).rolling(60,min_periods=40).mean();pt=persist*np.nan
for d in p.index:
 z=pd.concat([persist.loc[d].rename('y'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:pt.loc[d,z.index]=z.y-np.c_[np.ones(len(z)),z.t]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.t],z.y,rcond=None)[0]
short=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=8).corr(peer[a].where(peer[a]<0)) for a in A});change=short-downp;dep=change*np.nan
for d in p.index:
 z=pd.concat([change.loc[d].rename('y'),downp.loc[d].rename('d')],axis=1).dropna()
 if len(z)>=8:dep.loc[d,z.index]=z.y-np.c_[np.ones(len(z)),z.d]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.d],z.y,rcond=None)[0]
idres=pd.DataFrame({a:r[a]-beta(r[a],peer[a])*peer[a] for a in A});skew=-idres.rolling(40,min_periods=30).skew();isk=skew*np.nan
for d in p.index:
 z=pd.concat([skew.loc[d].rename('y'),es.loc[d].rename('es'),downp.loc[d].rename('dp'),kurt.loc[d].rename('k'),trend.loc[d].rename('tr')],axis=1).dropna()
 if len(z)>=8:isk.loc[d,z.index]=z.y-np.c_[np.ones(len(z)),z[['es','dp','k','tr']]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z[['es','dp','k','tr']]],z.y,rcond=None)[0]
lib={'ravmom':trend,'volnorm_reversal':rev,'downside_peer_correlation':downp,'return_autocorrelation':aut,'risk_adjusted_trend':trend,'orthogonal_acceleration':orth,'negative_spx_beta':spx,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'inverse_upside_peer_correlation':upinv,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'asymmetric_peer_beta_resilience':asym,'peer_downside_tail_persistence_residual':pt,'residual_downside_peer_dependence_change':dep,'inverse_idiosyncratic_skewness':isk}
mx=-1;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman') if len(z)>1 else np.nan;print('LIB',n,'rho',rho,'cells',len(z))
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
