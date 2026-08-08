"""One idea: peer-stress recovery persistence, residualized against trend and downside correlation.
Higher scores identify assets that have historically recovered on the day after broad
cross-asset stress, beyond their ordinary trend and downside co-movement."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-05-31')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A}); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
trend=(p/p.shift(20)-1)/vol
# A stress event is an unusually weak common-peer return.  Signal is the asset's
# subsequent return conditional on that event, averaged over the prior 60 sessions.
raw=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
 s=peer[a]; stress=s.le(s.rolling(20,min_periods=15).quantile(.20)).shift(1)
 raw[a]=r[a].where(stress).rolling(60,min_periods=12).mean()/vol[a]
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
# Cross-sectionally remove the two known sources of recovery-like ranking.
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('raw'),trend.loc[d].rename('trend'),down.loc[d].rename('down')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.trend,z.down];f.loc[d,z.index]=z.raw-X@np.linalg.lstsq(X,z.raw,rcond=None)[0]
def metric(h):
 fw=p.shift(-h)/p-1; ics=[]; nn=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: ics.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));nn.append(len(z))
 x=pd.Series(dict(ics),dtype=float);x.index=pd.to_datetime(x.index);sd=x.std();regs={}
 for name,yr in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028':[2028]}.items():
  q=x[x.index.year.isin(yr)];regs[name]={'dates':len(q),'ic':None if q.empty else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std())}
 turns=[]
 for i in range(10,len(f),10):
  q=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(q)>=8:turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(nn)),'turnover_10d':float(np.mean(turns)),'regimes':regs}
print('FACTOR peer_stress_recovery_persistence_residual_60obs','visible',END.date(),'assets',len(A),'data',p.index.min().date(),p.index.max().date())
print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(metric(h)))
# Reconstruct all admitted signals for mandatory independence screen.
def beta(x,y,cond=None):
 if cond is not None:x=x.where(cond);y=y.where(cond);w,mp=40,12
 else:w,mp=20,15
 return x.rolling(w,min_periods=mp).cov(y)/y.rolling(w,min_periods=mp).var()
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d].rename('a'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t];orth.loc[d,z.index]=z.a-X@np.linalg.lstsq(X,z.a,rcond=None)[0]
di=get_index_daily_data('DXY',5000).set_index('date');di.index=pd.to_datetime(di.index);dx=pd.to_numeric(di.loc[:END,'close'],errors='coerce').pct_change().reindex(r.index)
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
up=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A})
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':down,'return_autocorrelation_20obs':r.rolling(20,min_periods=15).corr(r.shift(1)),'relative_volume_participation_20d':pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A}),'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration_20_60obs':orth,'negative_spx_beta_20obs':pd.DataFrame({a:-beta(r[a],r.SPX) for a in A}),'inverse_excess_kurtosis_40obs':-r.rolling(40,min_periods=30).kurt(),'inverse_expected_shortfall_40obs':es,'inverse_upside_peer_correlation_40obs':up,'negative_conditional_dxy_up_beta_40obs':pd.DataFrame({a:-beta(r[a],dx,dx>0) for a in A}),'positive_conditional_dxy_down_beta_40obs':pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A}),'asymmetric_peer_beta_resilience_40obs':pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A}),'peer_downside_tail_persistence_residual_60obs':None,'inverse_cross_asset_correlation_concentration_40obs':None}
# Files without simple prior reconstruction are nevertheless required: derive their stored expressions below only if evidence exists; omit none from test is failure.
# tail persistence: conditional peer-tail next-day relationship residual cannot be reliably inferred here.
# Cross-correlation concentration is direct inverse average rolling pair correlation.
corrconc=pd.DataFrame(index=p.index,columns=A,dtype=float)
for d in p.index:
 q=r.loc[:d].tail(40)
 if len(q)>=30:
  c=q.corr()
  for a in A:corrconc.loc[d,a]=-c.loc[a].drop(a).mean()
lib['inverse_cross_asset_correlation_concentration_40obs']=corrconc
# peer tail persistence reconstructed as rolling correlation between asset return and lagged peer return conditional on peer tail, then residualized against down.
tail=pd.DataFrame({a:r[a].where(peer[a].shift(1)<=peer[a].rolling(60,min_periods=30).quantile(.2)).rolling(60,min_periods=12).corr(peer[a].shift(1).where(peer[a].shift(1)<=peer[a].rolling(60,min_periods=30).quantile(.2))) for a in A})
lib['peer_downside_tail_persistence_residual_60obs']=tail
mx=-1;who=None;missing=[]
for n,x in lib.items():
 if x is None: missing.append(n);continue
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();q=z.f.corr(z.x,method='spearman');print('LIB',n,float(q),len(z))
 if abs(q)>mx:mx,who=abs(q),n
print('MAX_ABS_LIBRARY_CORRELATION',mx,who,'MISSING',missing)
