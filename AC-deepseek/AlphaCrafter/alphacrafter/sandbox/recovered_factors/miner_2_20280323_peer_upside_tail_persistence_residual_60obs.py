"""One idea: peer-upside tail persistence residual (60 observations)."""
import pandas as pd,numpy as np,json,glob
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index); return d
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A});p=p.loc[:p.dropna(how='all').index.max()];r=p.pct_change();vol=r.rolling(20,min_periods=15).std();trend=(p/p.shift(20)-1)/vol
# Frequency of being in daily cross-asset top-return quintile, residualized against 20d trend.
th=r.quantile(.8,axis=1); event=r.ge(th,axis=0).astype(float).where(r.notna());raw=event.rolling(60,min_periods=45).mean();f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def metric(h):
 fw=p.shift(-h)/p-1;ic=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: ic.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(ic));sd=x.std();reg={}
 for n,yr in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028':[2028]}.items():
  q=x[x.index.year.isin(yr)];reg[n]={'dates':len(q),'ic':None if q.empty else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std())}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':reg}
print('FACTOR peer_upside_tail_persistence_residual_60obs','visible',p.index.max().date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date());print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(metric(h)))
# Reconstruct signals of every admitted library factor, including the March downside-tail factor.
def beta(x,y,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(40 if c is not None else 20,min_periods=12 if c is not None else 15).cov(y)/y.rolling(40 if c is not None else 20,min_periods=12 if c is not None else 15).var()
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A});rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d].rename('a'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t];orth.loc[d,z.index]=z.a-X@np.linalg.lstsq(X,z.a,rcond=None)[0]
di=get_index_daily_data('DXY',5000).set_index('date');di.index=pd.to_datetime(di.index);dx=pd.to_numeric(di.close,errors='coerce').pct_change().reindex(r.index)
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A});spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});kurt=-r.rolling(40,min_periods=30).kurt();aut=r.rolling(20,min_periods=15).corr(r.shift(1));down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});up=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A});du=pd.DataFrame({a:-beta(r[a],dx,dx>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A});ua=pd.DataFrame({a:r[a].where(r[a].shift(1)>0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)>0))-r[a].where(r[a].shift(1)<0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)<0)) for a in A});part=pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A})
# admitted downside-tail residual
lo=r.quantile(.2,axis=1);lr=r.le(lo,axis=0).astype(float).where(r.notna()).rolling(60,min_periods=45).mean();tail=lr*np.nan
for d in p.index:
 z=pd.concat([lr.loc[d].rename('y'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t];tail.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation':down,'return_autocorrelation':aut,'relative_volume_participation':part,'risk_adjusted_trend':trend,'orthogonal_acceleration':orth,'negative_spx_beta':spx,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'inverse_upside_peer_correlation':up,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'asymmetric_peer_beta_resilience':asym,'upside_minus_downside_return_autocorr':ua,'peer_downside_tail_persistence_residual':tail}
mx=0;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();q=z.f.corr(z.x,method='spearman');print('LIB',n,float(q),len(z))
 if abs(q)>mx:mx=abs(q);who=n
print('MAX',mx,who)
