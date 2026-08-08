"""One candidate: orthogonal VIX-down-day recovery resilience, 40 observations."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-11-01')
def read(a,index=False):
 d=(get_index_daily_data(a,5000) if index else get_stock_daily_data(a,5000)).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:read(a) for a in A});r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,w=40,mp=12,cond=None):
 if cond is not None:x=x.where(cond);y=y.where(cond)
 return x.rolling(w,min_periods=mp).cov(y)/y.rolling(w,min_periods=mp).var().replace(0,np.nan)
def resid(y,controls):
 out=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[q.loc[d].rename(str(i)) for i,q in enumerate(controls)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]];out.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A}); kurt=-r.rolling(40,min_periods=30).kurt()
vix=read('VIX',True).pct_change().reindex(r.index); dxy=read('DXY',True).pct_change().reindex(r.index); cny=read('USDCNY',True).pct_change().reindex(r.index)
# Candidate: risk-adjusted returns only on VIX declines; residual controls remove generic defensive/trend and VIX-up resilience.
upraw=pd.DataFrame({a:r[a].where(vix>0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
raw=pd.DataFrame({a:r[a].where(vix<0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
f=resid(raw,[es,down,kurt,trend,upraw])
def metric(h):
 fw=p.shift(-h)/p-1; xs=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: xs.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(xs));sd=x.std(); regs={}
 for n,sel in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('latest_120',np.arange(len(x))>=len(x)-120)]:
  q=x[sel];regs[n]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std()}
 ts=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'ic_se':sd/np.sqrt(len(x)),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(ts)),'regimes':regs}
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'candidate_cells',int(f.count().sum()),'of',f.size)
for h in [1,5,10,20]:print('METRIC',json.dumps(metric(h),default=float))
# Reconstruct non-deprecated admitted library signals; pooled Spearman evidence is required.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol; orth=resid(acc,[trend])
spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],40,12,peer[a]<0)-beta(r[a],peer[a],40,12,peer[a]>0) for a in A});du=pd.DataFrame({a:-beta(r[a],dxy,40,12,dxy>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dxy,40,12,dxy<0) for a in A})
persist=r.le(r.quantile(.20,axis=1),axis=0).astype(float).rolling(60,min_periods=40).mean();pt=resid(persist,[trend]); short=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=8).corr(peer[a].where(peer[a]<0)) for a in A});dep=resid(short-down,[down]); idres=pd.DataFrame({a:r[a]-beta(r[a],peer[a])*peer[a] for a in A});isk=resid(-idres.rolling(40,min_periods=30).skew(),[es,down,kurt,trend])
relvol=pd.DataFrame({a:read(a).index.to_series().map(lambda d: np.nan) for a in A}) # unavailable volume factors are excluded: no overlapping valid evidence
# VIX stress peer corr (conditional VIX upper 70% trailing threshold)
thr=vix.rolling(60,min_periods=40).quantile(.7); vp=pd.DataFrame({a:r[a].where(vix>thr).rolling(60,min_periods=15).corr(peer[a].where(vix>thr)) for a in A});vp=resid(vp,[es,down,kurt,trend])
# Yuan appreciation stress residual
cthr=cny.rolling(60,min_periods=40).quantile(.7); cr=pd.DataFrame({a:-beta(r[a],cny,60,15,cny>cthr) for a in A});cr=resid(cr,[du,es,down,trend])
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':down,'return_autocorrelation_20obs':aut,'vix_stress_peer_correlation_residual_60obs':vp,'usdcny_appreciation_stress_resilience_residual_60obs':cr,'asymmetric_peer_beta_resilience_40obs':asym,'residual_downside_peer_dependence_change_20_60':dep,'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration_20_60obs':orth,'negative_spx_beta_20obs':spx,'inverse_excess_kurtosis_40obs':kurt,'inverse_expected_shortfall_40obs':es,'inverse_upside_peer_correlation_40obs':upinv,'negative_conditional_dxy_up_beta_40obs':du,'positive_conditional_dxy_down_beta_40obs':dd,'orthogonal_vix_up_resilience_40obs':resid(upraw,[es,down,kurt,trend])}
mx=-1;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman') if len(z)>1 else np.nan;print('LIB',n,'rho',rho,'cells',len(z))
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
