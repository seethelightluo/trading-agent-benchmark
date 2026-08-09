"""One idea: downside-tail co-participation residual, 40 observations.
On severe peer-return days, score assets that avoid joining the cross-sectional tail;
residualize known expected-shortfall, downside-peer-correlation, and trend signals."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-06-14')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index)
 return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A});r=p.pct_change();vol=r.rolling(20,min_periods=15).std()
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
# Raw: negative conditional probability of an own bottom-quintile return on peer severe days.
# Daily tail cutoffs deliberately use only that day's cross section (no future data).
tailcut=r.quantile(.20,axis=1); severe=peer.le(peer.rolling(40,min_periods=20).quantile(.20))
raw=pd.DataFrame({a:-(r[a].le(tailcut)&severe[a]).astype(float).where(severe[a]).rolling(40,min_periods=12).mean() for a in A})
trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
downp=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),es.loc[d].rename('es'),downp.loc[d].rename('dp'),trend.loc[d].rename('tr')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['es','dp','tr']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def calc(h):
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);x.index=pd.to_datetime(x.index); sd=x.std()
 regs={}
 for n,ys in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028_ytd':[2028],'latest_120':None}.items():
  q=x.tail(120) if ys is None else x[x.index.year.isin(ys)]
  regs[n]={'dates':len(q),'ic':None if len(q)==0 else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std())}
 turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turns)),'regimes':regs}
print('CANDIDATE downside_tail_coparticipation_residual_40obs visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date())
print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(calc(h)))
# Reconstruct all non-deprecated library signals, including the 2028 peer-tail residual.
def beta(x,y,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 w=40 if c is not None else 20;m=12 if c is not None else 15
 return x.rolling(w,min_periods=m).cov(y)/y.rolling(w,min_periods=m).var()
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]];orth.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
di=get_index_daily_data('DXY',5000).set_index('date');di.index=pd.to_datetime(di.index);dx=pd.to_numeric(di.loc[:END,'close'],errors='coerce').pct_change().reindex(r.index)
spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});kurt=-r.rolling(40,min_periods=30).kurt();aut=r.rolling(20,min_periods=15).corr(r.shift(1));upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A});du=pd.DataFrame({a:-beta(r[a],dx,dx>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A});ua=pd.DataFrame({a:r[a].where(r[a].shift(1)>0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)>0))-r[a].where(r[a].shift(1)<0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)<0)) for a in A});part=pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A})
# miner_2 2028: tail-event persistence residualized cross-sectionally vs trend
qcut=r.quantile(.20,axis=1);persist=(r.le(qcut,axis=0)).astype(float).rolling(60,min_periods=40).mean();ptrend=trend;ptail=persist*np.nan
for d in p.index:
 z=pd.concat([persist.loc[d].rename('y'),ptrend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t];ptail.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib={'ravmom':trend,'volnorm_reversal':rev,'downside_peer_correlation':downp,'return_autorrelation':aut,'relative_volume_participation':part,'risk_adjusted_trend':trend,'orthogonal_acceleration':orth,'negative_spx_beta':spx,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'inverse_upside_peer_correlation':upinv,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'asymmetric_peer_beta_resilience':asym,'peer_downside_tail_persistence_residual':ptail}
mx=-1;who=None
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z))
 if not np.isfinite(rho):raise RuntimeError('Missing correlation '+n)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
