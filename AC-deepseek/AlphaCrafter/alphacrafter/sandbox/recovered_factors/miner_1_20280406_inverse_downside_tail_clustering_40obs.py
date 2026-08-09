"""One idea: inverse idiosyncratic downside-tail clustering in a 40-observation window."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-04-05')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A});r=p.pct_change();vol=r.rolling(20,min_periods=15).std();peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
scale=vol.shift(1);tail=(r < -1.5*scale).astype(float).where(scale.notna());f=-tail.rolling(40,min_periods=30).mean()
def calc(h):
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(q)>=8:out.append((d,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));ns.append(len(q))
 x=pd.Series(dict(out));sd=x.std();regs={}
 for n,ys in {'2026':[2026],'2027':[2027],'2028':[2028]}.items():
  q=x[x.index.year.isin(ys)];regs[n]={'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std())}
 turn=[]
 for i in range(10,len(f),10):
  q=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':regs}
print('CANDIDATE inverse_downside_tail_clustering_40obs visible',END.date(),'assets',len(A),'coverage',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(calc(h)))
# All current 17 effective signals reconstructed for complete Spearman screen.
def beta(x,y,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(40 if c is not None else 20,min_periods=12 if c is not None else 15).cov(y)/y.rolling(40 if c is not None else 20,min_periods=12 if c is not None else 15).var()
trend=(p/p.shift(20)-1)/vol;rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]];orth.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
di=get_index_daily_data('DXY',5000).set_index('date');di.index=pd.to_datetime(di.index);dx=pd.to_numeric(di.loc[:END,'close'],errors='coerce').pct_change().reindex(r.index)
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A});spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});kurt=-r.rolling(40,min_periods=30).kurt();aut=r.rolling(20,min_periods=15).corr(r.shift(1));downp=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A});du=pd.DataFrame({a:-beta(r[a],dx,dx>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A});ua=pd.DataFrame({a:r[a].where(r[a].shift(1)>0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)>0))-r[a].where(r[a].shift(1)<0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)<0)) for a in A});part=pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A})
threshold=r.quantile(.2,axis=1);peerraw=r.le(threshold,axis=0).astype(float).rolling(60,min_periods=45).mean();peerres=peerraw*np.nan
for d in p.index:
 z=pd.concat([peerraw.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]];peerres.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
invconc=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in range(39,len(p)):
 c=r.iloc[t-39:t+1].corr(min_periods=30)
 for a in A:
  v=c.loc[a].drop(a).abs().dropna()
  if len(v)>=8:invconc.loc[p.index[t],a]=-v.mean()
lib={'ravmom':trend,'volnorm_reversal':rev,'downside_peer_correlation':downp,'return_autocorrelation':aut,'relative_volume_participation':part,'risk_adjusted_trend':trend,'orthogonal_acceleration':orth,'negative_spx_beta':spx,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'inverse_upside_peer_correlation':upinv,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'asymmetric_peer_beta_resilience':asym,'upside_minus_downside_return_autocorr':ua,'peer_downside_tail_persistence_residual':peerres,'inverse_cross_asset_correlation_concentration':invconc}
mx=-1;who=None
for n,x in lib.items():
 q=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman');print('LIB',n,rho,len(q))
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
