"""One idea: medium-term (60d) risk-adjusted strength orthogonal to 20d trend.
A lower-turnover continuation sleeve, removing direct short/medium trend exposure."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-09-06')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A}); r=p.pct_change(); vol20=r.rolling(20,min_periods=15).std(); peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,c=None,w=40,m=12):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(w,min_periods=m).cov(y)/y.rolling(w,min_periods=m).var()
trend=(p/p.shift(20)-1)/vol20; raw=(p/p.shift(60)-1)/r.rolling(60,min_periods=40).std()
# Candidate: contemporaneous cross-sectional residual from its only direct parent.
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),trend.loc[d].rename('x')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.x];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
# Full admitted-library reconstruction for mandatory pooled-signal audit.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A}); auto=r.rolling(20,min_periods=16).corr(r.shift(1)); up=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A}); spx=pd.DataFrame({a:-beta(r[a],r.SPX,w=20,m=15) for a in A}); kurt=-r.rolling(40,min_periods=30).kurt(); es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol20[a] for a in A}); asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A});
dx=pd.to_numeric(get_index_daily_data('DXY',5000).set_index('date').assign(date=lambda x:pd.to_datetime(x.index)).loc[:END,'close'],errors='coerce').pct_change().reindex(r.index);du=pd.DataFrame({a:-beta(r[a],dx,dx>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A});acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol20;orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8: orth.loc[d,z.index]=z.iloc[:,0]-np.c_[np.ones(len(z)),z.iloc[:,1]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1]],z.iloc[:,0],rcond=None)[0]
corconc=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).corr(peer[a]).abs() for a in A}) # proxy differs from mean pairwise; audit conservatively labels it
idraw=pd.DataFrame({a:-((r[a]-beta(r[a],peer[a],w=20,m=15)*peer[a]).rolling(40,min_periods=30).skew()) for a in A}); idio=idraw*np.nan
controls=[es,down,kurt,trend]
for d in p.index:
 z=pd.concat([idraw.loc[d].rename('y')]+[x.loc[d].rename(str(i)) for i,x in enumerate(controls)],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1:]];idio.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
# dependence-change signal reconstructed before residual controls; included as a strict correlation surrogate
chg=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=12).corr(peer[a].where(peer[a]<0))-r[a].where(peer[a]<0).rolling(60,min_periods=20).corr(peer[a].where(peer[a]<0)) for a in A})
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':down,'return_autocorrelation_20obs':auto,'relative_volume_participation_20d':pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A}),'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration':orth,'negative_spx_beta':spx,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'inverse_upside_peer_correlation':up,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'asymmetric_peer_beta_resilience':asym,'inverse_cross_asset_correlation_concentration_proxy':corconc,'orthogonal_inverse_idiosyncratic_skewness':idio,'residual_downside_peer_dependence_change_proxy':chg}
def metrics(h):
 fw=p.shift(-h)/p-1; xs=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:xs.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(xs));turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 reg={}
 for k,q in {'2026':x[x.index.year==2026],'2027':x[x.index.year==2027],'2028_ytd':x[x.index.year==2028],'latest_120':x.tail(120)}.items():reg[k]={'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std()) if q.std()>0 else None}
 return {'h':h,'ic':float(x.mean()),'icir':float(x.mean()/x.std()),'hit':float((x>0).mean()),'dates':len(x),'mean_instruments':float(np.mean(ns)),'turnover10':float(np.mean(turn)),'regimes':reg}
print('CANDIDATE medium_term_strength_trend_residual_60obs','visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date());print('COVERAGE',int(f.count().sum()),f.size,float(f.count().sum()/f.size))
for h in (1,5,10,20):print('METRIC',json.dumps(metrics(h)))
miss=[];mx=-1;who=None
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z))
 if not np.isfinite(rho):miss.append(n)
 elif abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who,'MISSING',miss)
