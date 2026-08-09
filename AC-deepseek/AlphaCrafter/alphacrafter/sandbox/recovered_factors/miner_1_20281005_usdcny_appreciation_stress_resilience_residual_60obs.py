"""One candidate: USDCNY appreciation-stress resilience residual (60 observations).
Higher score is lower sensitivity to large daily USDCNY appreciation, net of broad dollar and risk controls."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-10-04')
def prices(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:prices(a) for a in A}); r=p.pct_change(); idx=p.index

def macro(sym):
 d=get_index_daily_data(sym,5000).set_index('date');d.index=pd.to_datetime(d.index)
 return pd.to_numeric(d.loc[:END,'close'],errors='coerce').pct_change().reindex(idx)
def beta(x,y,mask=None,w=60,minp=15):
 if mask is not None:x=x.where(mask);y=y.where(mask)
 return x.rolling(w,min_periods=minp).cov(y)/y.rolling(w,min_periods=minp).var().replace(0,np.nan)
cn=macro('USDCNY'); dx=macro('DXY'); vix=macro('VIX')
# Conditional sensitivity is estimated only from historically observed upper-tail USDCNY sessions.
shock=cn>cn.rolling(60,min_periods=30).quantile(.70)
raw=pd.DataFrame({a:-beta(r[a],cn,shock,60,15) for a in A})
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A}); vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
dxyup=pd.DataFrame({a:-beta(r[a],dx,dx>0,40,12) for a in A})
# Cross-sectionally residualize broad dollar exposure plus conventional defensive characteristics.
f=raw*np.nan
for d in idx:
 z=pd.concat([raw.loc[d].rename('y'),dxyup.loc[d].rename('du'),es.loc[d].rename('es'),down.loc[d].rename('down'),trend.loc[d].rename('trend')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['du','es','down','trend']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def metrics(h):
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals)); sd=x.std(); regs={}
 for name,mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('latest_120',np.arange(len(x))>=len(x)-120)]:
  q=x[mask];regs[name]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if len(q)>1 else np.nan,'hit':(q>0).mean() if len(q) else np.nan}
 turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'ic_se':sd/np.sqrt(len(x)),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turns)),'regimes':regs}
print('FACTOR usdcny_appreciation_stress_resilience_residual_60obs')
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'candidate_cells',int(f.count().sum()),'of',f.size,'coverage',f.count().sum()/f.size)
for h in [1,5,10,20]:print('METRIC',json.dumps(metrics(h),default=float))
# Independence check against every currently non-deprecated library factor, reconstructed from definitions.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=acc*np.nan
for d in idx:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8: orth.loc[d,z.index]=z.iloc[:,0]-np.c_[np.ones(len(z)),z.iloc[:,1]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1]],z.iloc[:,0],rcond=None)[0]
spx=pd.DataFrame({a:-beta(r[a],r.SPX,None,20,15) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));kurt=-r.rolling(40,min_periods=30).kurt();upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0,60,15)-beta(r[a],peer[a],peer[a]>0,60,15) for a in A})
short=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=8).corr(peer[a].where(peer[a]<0)) for a in A});change=short-down;dep=change*np.nan
for d in idx:
 z=pd.concat([change.loc[d].rename('y'),down.loc[d].rename('d')],axis=1).dropna()
 if len(z)>=8:dep.loc[d,z.index]=z.y-np.c_[np.ones(len(z)),z.d]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.d]],z.y,rcond=None)[0]
# Correct residual construction above without fragile slice shape.
for d in idx:
 z=pd.concat([change.loc[d].rename('y'),down.loc[d].rename('d')],axis=1).dropna()
 if len(z)>=8: dep.loc[d,z.index]=z.y-np.c_[np.ones(len(z)),z.d.values]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.d.values],z.y,rcond=None)[0]
idres=pd.DataFrame({a:r[a]-beta(r[a],peer[a],None,60,30)*peer[a] for a in A});skew=-idres.rolling(40,min_periods=30).skew();isk=skew*np.nan
for d in idx:
 z=pd.concat([skew.loc[d].rename('y'),es.loc[d].rename('es'),down.loc[d].rename('dp'),kurt.loc[d].rename('k'),trend.loc[d].rename('tr')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['es','dp','k','tr']]];isk.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':down,'return_autocorrelation_20obs':aut,'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration_20_60obs':orth,'negative_spx_beta_20obs':spx,'inverse_excess_kurtosis_40obs':kurt,'inverse_expected_shortfall_40obs':es,'inverse_upside_peer_correlation_40obs':upinv,'negative_conditional_dxy_up_beta_40obs':dxyup,'positive_conditional_dxy_down_beta_40obs':pd.DataFrame({a:beta(r[a],dx,dx<0,40,12) for a in A}),'asymmetric_peer_beta_resilience_40obs':asym,'residual_downside_peer_dependence_change_20_60':dep,'vix_stress_peer_correlation_residual_60obs':None,'inverse_idiosyncratic_skewness_40obs':isk}
# reconstruct last admitted VIX signal for binding comparison
vs=vix>vix.rolling(60,min_periods=30).quantile(.70); vr=pd.DataFrame({a:r[a].where(vs).rolling(60,min_periods=15).corr(peer[a].where(vs)) for a in A}); vf=vr*np.nan
for d in idx:
 z=pd.concat([vr.loc[d].rename('y'),es.loc[d].rename('es'),down.loc[d].rename('dp'),kurt.loc[d].rename('k'),trend.loc[d].rename('tr')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['es','dp','k','tr']]];vf.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib['vix_stress_peer_correlation_residual_60obs']=vf
mx=-1;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman') if len(z)>1 else np.nan
 print('LIB',n,'rho',rho,'cells',len(z))
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
