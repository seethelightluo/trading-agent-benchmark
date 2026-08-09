"""One idea: dispersion-conditioned medium-term relative strength residual (60d).
When cross-asset return dispersion is elevated, medium-term relative strength may be more informative;
residualize it each date against admitted signals to make the candidate independent."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-07-26')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A}); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,c=None,w=40,m=12):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(w,min_periods=m).cov(y)/y.rolling(w,min_periods=m).var()
trend=(p/p.shift(20)-1)/vol; mom60=(p/p.shift(60)-1)/r.rolling(60,min_periods=40).std()
# Regime multiplier is known contemporaneously: high cross-sectional 20d return dispersion.
disp=(p/p.shift(20)-1).std(axis=1); dz=(disp-disp.rolling(120,min_periods=80).mean())/disp.rolling(120,min_periods=80).std(); raw=mom60.mul(dz.clip(-2,2),axis=0)
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); downp=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A}); kurt=-r.rolling(40,min_periods=30).kurt(); es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A}); aut=r.rolling(20,min_periods=15).corr(r.shift(1)); upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A}); spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A})
di=get_index_daily_data('DXY',5000).set_index('date');di.index=pd.to_datetime(di.index);dx=pd.to_numeric(di.loc[:END,'close'],errors='coerce').pct_change().reindex(r.index);du=pd.DataFrame({a:-beta(r[a],dx,dx>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A})
acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol; orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8: orth.loc[d,z.index]=z.iloc[:,0]-np.c_[np.ones(len(z)),z.iloc[:,1]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1]],z.iloc[:,0],rcond=None)[0]
# Current admitted (non-deprecated) reconstructable signals, including duplicates only once.
lib={'ravmom':trend,'volnorm_reversal':rev,'downside_peer_correlation':downp,'return_autocorrelation':aut,'relative_volume_participation':pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A}),'risk_adjusted_trend':trend,'orthogonal_acceleration':orth,'negative_spx_beta':spx,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'inverse_upside_peer_correlation':upinv,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'asymmetric_peer_beta_resilience':asym}
# Datewise residualization against all available controls (minimum 8); this both avoids factor duplication and retains broad coverage.
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y')]+[x.loc[d].rename(n) for n,x in lib.items()],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1:].values]; f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def calc(h):
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));sd=x.std(); regs={}
 for n,yrs in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028_ytd':[2028],'latest_120':None}.items():
  q=x.tail(120) if yrs is None else x[x.index.year.isin(yrs)];regs[n]={'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std()) if len(q)>1 and q.std()>0 else None}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':regs}
print('CANDIDATE dispersion_conditioned_medium_term_strength_residual_60obs visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date());print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]: print('METRIC',json.dumps(calc(h)))
mx=-1;who=None; bad=[]
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z))
 if not np.isfinite(rho):bad.append(n)
 elif abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who,'MISSING',bad)
