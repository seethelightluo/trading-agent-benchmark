"""One candidate: orthogonal volume-confirmed idiosyncratic trend (20d).
Tests whether trend accompanied by unusual participation predicts cross-asset returns,
after stripping ordinary trend, reversal, standalone participation and tail/dependence risk."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-07-12')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A});r=p.pct_change();vol=r.rolling(20,min_periods=15).std()
v=pd.DataFrame({a:pd.to_numeric(rd(a).volume,errors='coerce') for a in A})
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(40 if c is not None else 20,min_periods=12 if c is not None else 15).cov(y)/y.rolling(40 if c is not None else 20,min_periods=12 if c is not None else 15).var()
trend=(p/p.shift(20)-1)/vol
part=np.log(v/v.rolling(20,min_periods=10).mean())
# participation-confirmed standardized 20d trend, then datewise residualization
raw=trend*part
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
downp=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
kurt=-r.rolling(40,min_periods=30).kurt()
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),trend.loc[d].rename('tr'),part.loc[d].rename('pa'),rev.loc[d].rename('rv'),es.loc[d].rename('es'),downp.loc[d].rename('dp'),kurt.loc[d].rename('ku')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['tr','pa','rv','es','dp','ku']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def calc(h):
 fw=p.shift(-h)/p-1; xs=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: xs.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(xs),dtype=float);x.index=pd.to_datetime(x.index);sd=x.std()
 regs={}
 for n,yr in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028_ytd':[2028],'latest_120':None}.items():
  q=x.tail(120) if yr is None else x[x.index.year.isin(yr)]
  regs[n]={'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std()) if len(q)>1 and q.std()>0 else None}
 ts=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(ts)),'regimes':regs}
print('CANDIDATE orthogonal_volume_confirmed_idiosyncratic_trend_20obs visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date())
print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(calc(h)))
# Reconstruct every effective library signal; duplicated trend definitions are retained as named files.
acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]];orth.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
di=get_index_daily_data('DXY',5000).set_index('date');di.index=pd.to_datetime(di.index);dx=pd.to_numeric(di.loc[:END,'close'],errors='coerce').pct_change().reindex(r.index)
spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A});du=pd.DataFrame({a:-beta(r[a],dx,dx>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A})
qcut=r.quantile(.20,axis=1);persist=(r.le(qcut,axis=0)).astype(float).rolling(60,min_periods=40).mean();ptail=persist*np.nan
for d in p.index:
 z=pd.concat([persist.loc[d].rename('y'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t];ptail.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
# dependence change: 20d minus 60d correlation on peer-down days
c20=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=8).corr(peer[a].where(peer[a]<0)) for a in A});c60=downp;depraw=c20-c60;dep=depraw*np.nan
for d in p.index:
 z=pd.concat([depraw.loc[d].rename('y'),downp.loc[d].rename('dp'),asym.loc[d].rename('as'),ptail.loc[d].rename('pt'),orth.loc[d].rename('or')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['dp','as','pt','or']]];dep.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib={'miner1_ravmom':trend,'miner1_reversal':rev,'miner1_downside_peer':downp,'miner1_autocorr':aut,'miner2_asym_resilience':asym,'miner2_tail_persistence':ptail,'miner2_dependence_change':dep,'miner3_volume':part,'miner3_trend':trend,'miner3_acceleration':orth,'miner3_spx_beta':spx,'miner3_kurtosis':kurt,'miner3_expected_shortfall':es,'miner3_upside_peer':upinv,'miner3_dxy_up':du,'miner3_dxy_down':dd}
mx=-1;who=None
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z))
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
