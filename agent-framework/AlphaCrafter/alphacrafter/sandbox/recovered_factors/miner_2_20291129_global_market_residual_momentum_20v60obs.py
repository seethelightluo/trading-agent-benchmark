import os,json,pandas as pd,numpy as np
from scipy.stats import spearmanr
# One idea: global-market residual momentum.  A 20d asset return is adjusted for
# its trailing 60d beta to the equal-weight cross-asset market, isolating idiosyncratic relative strength.
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2029-11-28')
C={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); C[a]=d[d.date<=CUT].set_index('date').close
close=pd.DataFrame(C).sort_index(); r=close.pct_change(); market=r.mean(axis=1,skipna=True)
beta=r.rolling(60,min_periods=45).cov(market).div(market.rolling(60,min_periods=45).var(),axis=0)
sig=(close/close.shift(20)-1).sub(beta.mul(close.mean(axis=1)/close.mean(axis=1).shift(20)-1,axis=0)).replace([np.inf,-np.inf],np.nan)
def stat(s,h):
 f=close.shift(-h)/close-1; z=[];n=[]
 for t in s.index:
  m=s.loc[t].notna()&f.loc[t].notna()
  if m.sum()>=8:z.append(spearmanr(s.loc[t,m],f.loc[t,m]).statistic);n.append(m.sum())
 z=np.array(z); sd=z.std(ddof=1)
 return {'ic_dates':len(z),'mean_valid_instruments':float(np.mean(n)),'ic':float(z.mean()),'icir':float(z.mean()/sd),'hit_ratio':float((z>0).mean()),'se':float(sd/np.sqrt(len(z)))}
print('IDEA global_market_residual_momentum_20v60');print('cutoff',CUT.date(),'panel_dates',len(sig),'coverage',round(float(sig.notna().mean().mean()),5))
for h in [1,5,10,20]: print('H',h,{k:round(v,6) for k,v in stat(sig,h).items()})
rr=[]
for i in range(1,len(sig)):
 m=sig.iloc[i].notna()&sig.iloc[i-1].notna()
 if m.sum()>=8:rr.append(spearmanr(sig.iloc[i,m],sig.iloc[i-1,m]).statistic)
print('rank_stability',round(float(np.nanmean(rr)),6),'turnover_proxy',round(float(1-np.nanmean(rr)),6))
for nm,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029','2029-01-01','2029-11-28')]:print('REG',nm,{k:round(v,6) for k,v in stat(sig.loc[lo:hi],5).items()})
# exact signal artifacts for all currently effective factors (curated, complete audit)
paths={'downside_concentration_continuation_10v40obs':'scripts/miner_2_20271118_downside_concentration_continuation_10v40obs_signal.pkl','drawdown_velocity_reversal_60d':'scripts/miner_3_20270408_drawdown_velocity_reversal_60d_signal.pkl','miner_2_volume_confirmed_drawdown_recovery_60d':'scripts/miner_2_20261105_volume_confirmed_drawdown_recovery_60d_signal.pkl','miner_3_relative_volume_participation_20d':'scripts/miner_3_20260716_relative_volume_participation_20d_signal.pkl','miner_1_semivolatility_balance_improvement_10d':'scripts/miner_1_20261217_semivolatility_balance_improvement_10d_signal.pkl','miner_2_range_compressed_intermediate_continuation_10to20x10v40obs':'scripts/miner_2_20280504_range_compressed_intermediate_continuation_10to20x10v40obs_signal.pkl','miner_1_inverse_directional_recovery_efficiency_10d':'scripts/miner_1_20270603_inverse_directional_recovery_efficiency_10d_signal.pkl','miner_3_risk_adjusted_trend_20d':'scripts/miner_3_20260716_risk_adjusted_trend_20d_signal.pkl','inverse_return_serial_dependence_20obs':'scripts/miner_2_20270701_inverse_return_serial_dependence_20obs_signal.pkl','drawdown_scaled_downside_streak_exhaustion_10x40obs':'scripts/miner_1_20280323_drawdown_scaled_downside_streak_exhaustion_10x40obs_signal.pkl','miner_1_directional_volume_imbalance_30obs':'scripts/miner_1_20270909_directional_volume_imbalance_30obs_signal.pkl','upside_concentration_exhaustion_10v40obs':'scripts/miner_2_20271202_upside_concentration_exhaustion_10v40obs_signal.pkl','miner_1_downside_upside_volatility_balance_20d':'scripts/miner_1_20261203_downside_upside_volatility_balance_20d_signal.pkl','miner_3_vix_normalization_downside_skew_reversal_20d':'scripts/miner_3_20271230_vix_normalization_downside_skew_reversal_20d_signal.pkl','miner_2_normalized_overnight_gap_reversal_5v20obs':'scripts/miner_2_20280824_normalized_overnight_gap_reversal_5v20obs_signal.pkl','miner_3_vix_normalization_rebound_reversal_5d':'scripts/miner_3_20271202_vix_normalization_rebound_reversal_5d_signal.pkl','miner_1_inverted_downside_cross_asset_beta_40d':'scripts/miner_1_20270114_inverted_downside_cross_asset_beta_40d_signal.pkl','miner_3_vix_shock_resilience_20d':'scripts/miner_3_20260827_vix_shock_resilience_20d_signal.pkl','miner_2_standardized_jump_asymmetry_20v40obs':'scripts/miner_2_20280113_standardized_jump_asymmetry_20v40obs_signal.pkl','miner_2_realized_volatility_20obs':'scripts/miner_2_20260716_realized_volatility_20obs_signal.pkl','post_recovery_reversal_20d':'scripts/miner_3_20270715_post_recovery_reversal_20d_signal.pkl','miner_1_volnorm_reversal_5obs':'scripts/miner_1_20260716_volnorm_reversal5_signal.pkl','miner_1_cross_asset_trend_rank_acceleration_10v60obs':'scripts/miner_1_20280504_cross_asset_trend_rank_acceleration_10v60obs_signal.pkl'}
res=[];missing=[]
for fid,p in paths.items():
 try:
  x=pd.read_pickle(p); x.index=pd.to_datetime(x.index);x=x.reindex(index=sig.index,columns=A);q=[]
  for t in sig.index:
   m=sig.loc[t].notna()&x.loc[t].notna()
   if m.sum()>=8:q.append(spearmanr(sig.loc[t,m],x.loc[t,m]).statistic)
  if q:res.append((fid,float(np.max(np.abs(q))),len(q)))
  else:missing.append(fid)
 except Exception as e:missing.append(fid)
res.sort(key=lambda x:-x[1]);print('AUDIT compared',len(res),'missing',missing);print('TOP_CORR',res[:8]);print('MAX_ABS_LIBRARY_CORR',res[0][1] if len(res)==len(paths) else 'UNAVAILABLE')
pd.to_pickle(sig,'scripts/miner_2_20291129_global_market_residual_momentum_20v60obs_signal.pkl')
