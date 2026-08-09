"""Validate one factor: DXY-strength-conditioned peer-relative reversal (60 sessions), with full active-library novelty audit."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']
def close(sym,index=False):
 d=(get_index_daily_data(sym,5000) if index else get_stock_daily_data(sym,5000)).copy(); d.date=pd.to_datetime(d.date)
 return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame({a:close(a) for a in A}).sort_index(); r=P.pct_change(); m=r.median(1); rel=r.sub(m,axis=0); cs=lambda x:x.sub(x.median(1),axis=0)
# Candidate: on days where the lagged five-session DXY trend is positive, accumulate each asset's peer-relative return and reverse it.
# DXY is an observation-only conditioning series and is never included in the cross-section.
DXY=close('DXY',True).reindex(P.index).pct_change(); dxy_up=DXY.rolling(5,min_periods=4).sum().shift(1)>0
F=cs(-rel.where(dxy_up,axis=0).rolling(60,min_periods=12).mean().shift(1))
cut=P.index.max(); print('FACTOR dxy_strength_conditioned_peer_relative_reversal_60 cutoff',cut.date(),'assets',len(A))
def summary(x,h,span=None):
 if span:x=x.loc[span[0]:span[1]]
 y=(P.shift(-h)/P-1).reindex(x.index); z=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z); return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),4),'mean_n':round(np.mean(ns),2),'min_n':min(ns)}
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(F.notna().stack().mean(),6),'DXY_UP_DATES',int(dxy_up.sum()))
for h in [1,5,10,20]:print('H',h,summary(F,h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-12-31')),('2029_current',('2029-01-01',str(cut.date()))),('recent180',(str(cut-pd.Timedelta(days=180)),str(cut.date())))]:print('REGIME10',n,summary(F,10,s))
print('TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CROSS_SECTIONAL_SD',round(F.std(1).mean(),6))
# faithful full admitted-library reconstructions
neg=r.where(r<0,0); vol20=r.rolling(20,min_periods=15).std(); peak=P.rolling(60,min_periods=45).max(); dd=P/peak-1
beta=lambda x,y,w:x.rolling(w,min_periods=max(12,w//2)).cov(y)/y.rolling(w,min_periods=max(12,w//2)).var()
V=pd.DataFrame({a:pd.to_numeric(get_stock_daily_data(a,5000).sort_values('date').set_index(pd.to_datetime(get_stock_daily_data(a,5000).sort_values('date').date)).volume,errors='coerce') for a in A}).reindex(P.index)
rv=np.log(V/V.rolling(20,min_periods=12).mean()); VIX=close('VIX',True).reindex(P.index).pct_change()
lib={}
lib['ravmom_20obs']=(P/P.shift(20)-1)/vol20
lib['volnorm_reversal_5obs']=-(P/P.shift(5)-1)/r.rolling(5,min_periods=4).std()
lib['vix_regime_conditioned_risk_adjusted_trend_20']=((P/P.shift(20)-1)/vol20).mul(np.where((VIX.rolling(1).sum()/VIX.shift(20).rolling(1).sum()-1)>0,-1,1),axis=0)
lib['downside_cross_asset_beta_resilience_40']=pd.DataFrame({a:beta(r[a].where(m<0),m.where(m<0),40) for a in A})
lib['inverse_idiosyncratic_volatility_20']=-(rel.rolling(20,min_periods=15).std());lib['stable_liquidity_participation_20']=-rv.rolling(20,min_periods=15).std();lib['return_skewness_60']=r.rolling(60,min_periods=40).skew()
lib['gradual_volatility_contraction_gated_trend_20']=((P/P.shift(20)-1)/vol20)*np.tanh(np.clip(-np.log(vol20/r.rolling(40,min_periods=20).std()),-2,2))
lib['dxy_directional_return_asymmetry_60']=pd.DataFrame({a:r[a].where(DXY<0).rolling(60,min_periods=35).mean()-r[a].where(DXY>0).rolling(60,min_periods=35).mean() for a in A})
lib['post_stress_relative_rebound_reversal_60']=cs(-rel.where(m.shift(1)<m.shift(1).rolling(60,min_periods=35).quantile(.25)).rolling(60,min_periods=12).mean())
lib['delayed_post_stress_relative_rebound_reversal_60']=cs(-((P/P.shift(3)-1).sub((P/P.shift(3)-1).median(1),axis=0)).where(m.shift(5)<m.shift(5).rolling(60,min_periods=35).quantile(.25)).rolling(60,min_periods=12).mean())
common=pd.DataFrame({a:r[a].rolling(20,min_periods=12).corr(r.drop(columns=a).median(1)) for a in A}).median(1); ce=(common>common.rolling(60,min_periods=35).quantile(.75))&(common>common.shift(5))
lib['commonality_shock_peer_relative_reversal_60']=cs(-((P/P.shift(5)-1).sub((P/P.shift(5)-1).median(1),axis=0)/vol20).where(ce,axis=0).rolling(60,min_periods=1).mean())
ds=rel.std(1); de=(ds>ds.rolling(60,min_periods=35).quantile(.75))&(ds>ds.shift(5));lib['dispersion_shock_peer_reversal_20']=cs(-(rel.rolling(5).sum().div(vol20)).where(de,axis=0).rolling(20,min_periods=5).mean())
lib['inverse_volume_weighted_peer_tail_asymmetry_60']=cs(-((rel.clip(lower=0)*rv.rank(pct=True)).rolling(60,min_periods=40).sum()-((-rel).clip(lower=0)*rv.rank(pct=True)).rolling(60,min_periods=40).sum())/((rel.abs()*rv.rank(pct=True)).rolling(60,min_periods=40).sum()))
lib['stress_duration_weighted_peer_resilience_reversal_60']=cs(-(rel.where(m<-.35*m.rolling(60,min_periods=30).std(),axis=0)*(1+.25*(m<-.35*m.rolling(60,min_periods=30).std()).rolling(5).sum().shift(1))).rolling(60,min_periods=5).mean().shift(1))
lib['relative_volume_participation_20d']=rv;lib['risk_adjusted_trend_20d']=(P/P.shift(20)-1)/vol20;lib['quiet_trend_path_efficiency_20_60']=(P/P.shift(20)-1).abs()/r.abs().rolling(20).sum()*(1-r.rolling(20).std().rank(pct=True));lib['vix_upside_shock_beta_resilience_40']=pd.DataFrame({a:-beta(r[a].where(VIX>0),VIX.where(VIX>0),40) for a in A})
lib['downside_event_excess_magnitude_median_40']=cs(rel.where(m<m.rolling(60,min_periods=35).quantile(.35),axis=0).rolling(40,min_periods=12).median());lib['inverse_lower_tail_persistence_40_60']=-pd.DataFrame({a:(r[a]<r[a].shift(1).rolling(60,min_periods=40).quantile(.2)).rolling(40,min_periods=25).mean() for a in A});lib['upside_return_concentration_60']=r.clip(lower=0).rolling(60,min_periods=40).max()/r.clip(lower=0).rolling(60,min_periods=40).sum();lib['smooth_peer_relative_drawdown_recovery_60_10']=cs((dd-dd.shift(10))/(.01-dd.shift(10)));lib['peer_relative_downside_volatility_compression_10_40']=cs(-np.log((np.sqrt((neg**2).rolling(10,min_periods=7).mean())+1e-5)/(np.sqrt((neg**2).rolling(40,min_periods=25).mean())+1e-5)));lib['downside_correlation_regime_spread_20_80']=cs(pd.DataFrame({a:beta(r[a].where(m<0),m.where(m<0),20)-beta(r[a].where(m<0),m.where(m<0),80) for a in A}))
mx=0;who='';valid=0
for n,g in lib.items():
 q=pd.concat([F.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else None)
 if np.isfinite(rho):valid+=1; mx,who=(abs(rho),n) if abs(rho)>mx else (mx,who)
print('WHOLE_LIBRARY',len(lib),'reconstructed',valid,'MAX_ABS',round(float(mx),6),who)
