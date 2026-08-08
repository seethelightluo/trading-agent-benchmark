"""One candidate: idiosyncratic volatility compression continuation, through 2028-11-30.
Score is negative ratio of 10d realized volatility to its 60d realized volatility.
Assets whose current volatility has compressed versus their own recent baseline may
attract trend-following allocation and exhibit near-term relative persistence.
Uses only closes at each signal date; no future data.
"""
import glob,json,os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-11-30')
def close(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,'close'].astype(float)
P=pd.DataFrame({a:close(a) for a in A}).sort_index(); r=np.log(P/P.shift())
vol10=r.rolling(10,min_periods=8).std(); vol60=r.rolling(60,min_periods=45).std()
F=-(vol10/vol60.replace(0,np.nan))
def met(h):
 fw=P.shift(-h)/P-1; rows=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   rows.append((d,float(spearmanr(z.f,z.r).statistic)));ns.append(len(z))
 s=pd.Series(dict(rows),dtype=float); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
allmet={}
for h in (1,5,10,20):
 s,m=met(h);allmet[h]=m;print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==5:
  for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2025',s.index.year.isin([2024,2025])),('2026_2028',s.index.year>=2026)]:
   q=s[mask];print('REGIME_5D',lab,json.dumps({'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std(ddof=1)),'hit':float((q>0).mean())}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('PANEL',json.dumps({'signal_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'implied_turnover':float(1-np.mean(st))}))
F.to_pickle('scripts/miner_2_20281130_volatility_compression_ratio_10v60obs_signal.pkl')
# audit every currently effective factor using persisted research artifacts
alias={
'miner_2_realized_volatility_20obs':'miner_2_20260716_realized_volatility_20obs_signal.pkl','miner_2_volume_confirmed_drawdown_recovery_60d':'miner_2_20261105_volume_confirmed_drawdown_recovery_60d_signal.pkl','inverse_return_serial_dependence_20obs':'miner_2_20270701_inverse_return_serial_dependence_20obs_signal.pkl','downside_concentration_continuation_10v40obs':'miner_2_20271118_downside_concentration_continuation_10v40obs_signal.pkl','upside_concentration_exhaustion_10v40obs':'miner_2_20271202_upside_concentration_exhaustion_10v40obs_signal.pkl','miner_2_standardized_jump_asymmetry_20v40obs':'miner_2_20280113_standardized_jump_asymmetry_20v40obs_signal.pkl','miner_2_range_compressed_intermediate_continuation_10to20x10v40obs':'miner_2_20280504_range_compressed_intermediate_continuation_10to20x10v40obs_signal.pkl','miner_2_normalized_overnight_gap_reversal_5v20obs':'miner_2_20280824_normalized_overnight_gap_reversal_5v20obs_signal.pkl',
'miner_1_downside_upside_volatility_balance_20d':'miner_1_20261203_downside_upside_volatility_balance_20d_signal.pkl','miner_1_semivolatility_balance_improvement_10d':'miner_1_20261217_semivolatility_balance_improvement_10d_signal.pkl','miner_1_inverted_downside_cross_asset_beta_40d':'miner_1_20270114_inverted_downside_cross_asset_beta_40d_signal.pkl','miner_1_inverse_directional_recovery_efficiency_10d':'miner_1_20270603_inverse_directional_recovery_efficiency_10d_signal.pkl','miner_1_directional_volume_imbalance_30obs':'miner_1_20270909_directional_volume_imbalance_30obs_signal.pkl','drawdown_scaled_downside_streak_exhaustion_10x40obs':'miner_1_20280323_drawdown_scaled_downside_streak_exhaustion_10x40obs_signal.pkl','miner_1_cross_asset_trend_rank_acceleration_10v60obs':'miner_1_20280504_cross_asset_trend_rank_acceleration_10v60obs_signal.pkl','miner_1_volnorm_reversal_5obs':'miner_1_20260716_volnorm_reversal5_signal.pkl',
'miner_3_relative_volume_participation_20d':'miner_3_20260716_relative_volume_participation_20d_signal.pkl','miner_3_risk_adjusted_trend_20d':'miner_3_20260716_risk_adjusted_trend_20d_signal.pkl','miner_3_vix_shock_resilience_20d':'miner_3_20260827_vix_shock_resilience_20d_signal.pkl','drawdown_velocity_reversal_60d':'miner_3_20270408_drawdown_velocity_reversal_60d_signal.pkl','post_recovery_reversal_20d':'miner_3_20270715_post_recovery_reversal_20d_signal.pkl','miner_3_vix_normalization_rebound_reversal_5d':'miner_3_20271202_vix_normalization_rebound_reversal_5d_signal.pkl','miner_3_vix_normalization_downside_skew_reversal_20d':'miner_3_20271230_vix_normalization_downside_skew_reversal_20d_signal.pkl'}
rows=[];missing=[]
for path in glob.glob('factors/*.json'):
 if path.endswith('.bak'):continue
 d=json.load(open(path))
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 fid=d['factor_id'];fn=alias.get(fid)
 if not fn or not os.path.exists('scripts/'+fn):missing.append(fid);continue
 L=pd.read_pickle('scripts/'+fn);x,y=F.align(L,join='inner',axis=0);z=pd.DataFrame({'x':x.stack(),'y':y.stack()}).dropna()
 rho=float(spearmanr(z.x,z.y).statistic) if len(z)>2 else float('nan');rows.append((fid,len(z),rho));print('LIBRARY_CORR',fid,len(z),rho)
print('LIBRARY_MISSING',json.dumps(missing))
if not missing and rows:
 b=max(rows,key=lambda q:abs(q[2]));print('LIBRARY_MAX',json.dumps({'factor':b[0],'cells':b[1],'rho':b[2],'max_abs_library_correlation':abs(b[2]),'audited_factors':len(rows)}))
