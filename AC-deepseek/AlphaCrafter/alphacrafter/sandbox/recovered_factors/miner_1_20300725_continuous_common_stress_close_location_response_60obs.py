"""Single idea: continuous common-stress close-location response, 60 observations."""
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2030-07-24')
def ld(a,c='close',idx=False):
 d='../persistent/index_data/' if idx else '../persistent/stock_data/'
 return pd.read_csv(d+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,c].astype(float)
def beta(x,y,w=60,cond=None):
 if isinstance(x,pd.DataFrame): return pd.DataFrame({a:beta(x[a],y,w,cond[a] if isinstance(cond,pd.DataFrame) else cond) for a in x})
 yy=y.reindex(x.index,method='ffill'); cc=cond.reindex(x.index).fillna(False) if cond is not None else None; xx=x.where(cc) if cc is not None else x; yy=yy.where(cc) if cc is not None else yy
 return xx.rolling(w,min_periods=12).cov(yy).div(yy.rolling(w,min_periods=12).var())
p=pd.DataFrame({a:ld(a) for a in A});r=p.pct_change(fill_method=None);hi=pd.DataFrame({a:ld(a,'high') for a in A});lo=pd.DataFrame({a:ld(a,'low') for a in A});V=pd.DataFrame({a:ld(a,'volume') for a in A})
loc=(p-lo).div((hi-lo).replace(0,np.nan));m=r.median(axis=1); stress=(-m.shift()/m.rolling(60,min_periods=30).std()).clip(-4,4)
f=pd.DataFrame({a:loc[a].rolling(60,min_periods=30).cov(stress).div(stress.rolling(60,min_periods=30).var()) for a in A})
print('CANDIDATE continuous_common_stress_close_location_response_60obs cutoff',END.date(),'assets',len(A));print('expression=rolling_60_beta(close_location[t], -median_crossasset_return[t-1]/rolling_std_60(median_return)[t-1])')
best=None
for h in [1,5,10,20]:
 y=p.shift(-h).div(p).sub(1);z=[];cv=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:z.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));cv.append(len(q)/15)
 x=pd.Series(dict(z));ic=x.mean();ir=ic/x.std(ddof=1)
 print(f'H={h} dates={len(x)} IC={ic:.6f} ICIR={ir:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cv):.4f} mean_instruments={15*np.mean(cv):.2f} GATE={abs(ic)>=.007 and abs(ir)>=.084}')
 if len(x) and (best is None or abs(ic*ir)>abs(best[1].mean()*(best[1].mean()/best[1].std(ddof=1)))):best=(h,x)
h,x=best;print('BEST_HORIZON',h)
for n,l,u in [('2020-21','2020','2022'),('2022-23','2022','2024'),('2024-25','2024','2026'),('2026-current','2026','2031')]:
 z=x[(x.index>=l)&(x.index<u)];print(f'REGIME {n} dates={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}; concentration_mean_sd={f.std(axis=1).mean():.6f}')
# Reconstruct every presently admitted signal using the same completed data.
v=r.rolling(20,min_periods=15).std();v60=r.rolling(60,min_periods=45).std();vix=ld('VIX',idx=True).pct_change();dxy=ld('DXY',idx=True).pct_change();neg=r<0
ca=pd.DataFrame({a:r[a].where(m<0).rolling(60,min_periods=12).corr(m.where(m<0))-r[a].where(m>=0).rolling(60,min_periods=12).corr(m.where(m>=0)) for a in A});b=beta(r,m);db=beta(r,m,60,m<0)-b;bo=pd.DataFrame(index=r.index,columns=A);ex=bo.copy()
for t in r.index:
 q=pd.concat([b.loc[t],v.loc[t]],axis=1).dropna()
 if len(q)>=3:z=np.polyfit(q.iloc[:,1],q.iloc[:,0],1);bo.loc[t]=b.loc[t]-(z[1]+z[0]*v.loc[t])
 q=pd.concat([db.loc[t],ca.loc[t]],axis=1).dropna()
 if len(q)>=3:z=np.polyfit(q.iloc[:,1],q.iloc[:,0],1);ex.loc[t]=db.loc[t]-(z[1]+z[0]*ca.loc[t])
d20=np.log(V.where(neg).rolling(20,min_periods=5).mean()/V.where(~neg).rolling(20,min_periods=5).mean());d60=np.log(V.where(neg).rolling(60,min_periods=12).mean()/V.where(~neg).rolling(60,min_periods=12).mean());rng=(hi-lo).abs()/p;cs=v.median(axis=1);res=r-b.mul(m,axis=0)
lib={'realized_volatility_20obs':v,'volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'correlation_asymmetry_60obs':ca,'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean()-.5,'dispersion_sensitivity_20obs':pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(r.std(axis=1)) for a in A}),'volatility_clustering_autocorr_20obs':r.abs().rolling(20,min_periods=15).corr(r.abs().shift()),'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).corr(r.shift()),'return_directional_efficiency_20obs':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),'relative_liquidity_stress_20_60obs':np.log((r.abs()/V).rolling(20,min_periods=15).mean()/(r.abs()/V).rolling(60,min_periods=45).mean()),'risk_adjusted_trend_20d':(p/p.shift(20)-1)/v,'risk_adjusted_trend_acceleration_20_60d':(p/p.shift(20)-1)/v-(p/p.shift(60)-1)/v60,'relative_volume_participation_20d':np.log(V/V.rolling(20,min_periods=15).mean()),'adaptive_vix_relief_beta_change_25_60obs':beta(r,vix,25,vix<0)-beta(r,vix,60,vix<0),'vix_shock_relief_beta_asymmetry_60obs':beta(r,vix,60,vix>0)-beta(r,vix,60,vix<0),'dxy_median_trend_regime_beta_spread_60obs':beta(r,dxy,60,m.rolling(20,min_periods=15).median()>0)-beta(r,dxy,60,m.rolling(20,min_periods=15).median()<=0),'downside_volume_participation_asymmetry_60obs':d60,'inverted_downside_volume_participation_acceleration_20_60obs':-(d20-d60),'vol_orthogonal_median_beta_60obs':bo,'excess_downside_beta_ca_orthogonal_60obs':ex,'residual_downside_semivol_share_60obs':np.sqrt(res.clip(upper=0).pow(2).rolling(60,min_periods=45).mean())/np.sqrt(res.pow(2).rolling(60,min_periods=45).mean()),'dxy_relative_vol_regime_beta_spread_60obs':beta(r,dxy,60,v.gt(v.rolling(60,min_periods=45).median()))-beta(r,dxy,60,v.le(v.rolling(60,min_periods=45).median())),'inverted_dispersion_regime_range_state_20_60obs':-np.log(rng.rolling(20,min_periods=15).mean()/rng.rolling(60,min_periods=45).mean()).mul(np.sign(np.log(cs/cs.rolling(60,min_periods=45).median())).replace(0,1),axis=0),'dxy_shock_lagged_response_persistence_60obs':beta(r,dxy,60,dxy.abs()>dxy.abs().rolling(60,min_periods=30).median()),'vix_tail_lagged_response_persistence_60obs':beta(r,vix,60,vix.abs()>vix.abs().rolling(60,min_periods=30).median())}
mx=-1;missing=[]
for n,o in lib.items():
 q=pd.concat([f.stack(),o.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)==0:missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman');print(f'LIBRARY {n} rho={rho:.6f} cells={len(q)}')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print(f'max_abs_library_correlation={mx:.6f}; closest={who}; closest_cells={cells}; signals_tested={len(lib)}; missing={missing}')
f.to_pickle('scripts/miner_1_20300725_continuous_common_stress_close_location_response_60obs_candidate_signal.pkl')
