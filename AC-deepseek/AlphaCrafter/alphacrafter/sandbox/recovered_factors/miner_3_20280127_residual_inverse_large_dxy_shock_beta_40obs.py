"""One idea: residual inverse beta to large absolute DXY moves (40 observations)."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-01-26')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A});r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
di=get_index_daily_data('DXY',5000).set_index('date');di.index=pd.to_datetime(di.index); dx=pd.to_numeric(di.loc[:END,'close'],errors='coerce').pct_change().reindex(r.index)
def beta(x,y,c,minp=12): return x.where(c).rolling(40,min_periods=minp).cov(y.where(c))/y.where(c).rolling(40,min_periods=minp).var()
# Exposure when DXY makes a large move relative to its trailing 20d distribution; negative sign favors assets resilient to broad USD shocks.
shock=dx.abs()>dx.abs().rolling(20,min_periods=15).median(); raw=pd.DataFrame({a:-beta(r[a],dx,shock) for a in A})
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
du=pd.DataFrame({a:-beta(r[a],dx,dx>0) for a in A}); dd=pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A})
# Cross-sectional residual makes this a distinct large-shock response, not ordinary directional-DXY beta or short reversal/tail risk.
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),du.loc[d].rename('u'),dd.loc[d].rename('dn'),rev.loc[d].rename('rv'),es.loc[d].rename('es')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['u','dn','rv','es']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def metric(h):
 fw=p.shift(-h)/p-1; out=[]; nums=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));nums.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(); reg={}
 for n,yrs in {'2020_2021':[2020,2021],'2022_2023':[2022,2023],'2024_2025':[2024,2025],'2026':[2026],'2027':[2027],'2028':[2028]}.items():
  q=x[x.index.year.isin(yrs)];reg[n]={'dates':len(q),'ic':None if len(q)==0 else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std()),'hit':None if len(q)==0 else float((q>0).mean())}
 t=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:t.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(nums)),'turnover_10d':float(np.mean(t)),'regimes':reg}
print('FACTOR residual_inverse_large_dxy_shock_beta_40obs','visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date());print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(metric(h)))
# admitted library signal reconstruction
trend=(p/p.shift(20)-1)/vol; acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]];orth.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
spxb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(r.SPX)/r.SPX.rolling(20,min_periods=15).var() for a in A});kurt=-r.rolling(40,min_periods=30).kurt();autoc=r.rolling(20,min_periods=15).corr(r.shift(1));down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A}); invup=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A}); asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A});part=pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A})
# Up-minus-down own return autocorrelation, matching admitted miner_2 definition.
ua=pd.DataFrame({a:r[a].where(r[a].shift(1)>0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)>0))-r[a].where(r[a].shift(1)<0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)<0)) for a in A})
lib={'ravmom':trend,'volnorm_reversal':rev,'downside_peer_correlation':down,'return_autocorrelation':autoc,'relative_volume_participation':part,'risk_adjusted_trend':trend,'orthogonal_acceleration':orth,'negative_spx_beta':spxb,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'inverse_upside_peer_correlation':invup,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'asymmetric_peer_beta_resilience':asym,'upside_minus_downside_return_autocorr':ua}
mx=-1
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); q=z.f.corr(z.x,method='spearman');print('LIB',n,round(q,6),len(z))
 if not np.isfinite(q): raise RuntimeError('missing mandatory correlation '+n)
 if abs(q)>mx:mx=abs(q);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who)
