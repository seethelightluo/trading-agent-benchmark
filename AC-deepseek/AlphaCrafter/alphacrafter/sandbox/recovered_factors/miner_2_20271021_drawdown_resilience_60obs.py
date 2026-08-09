"""miner_2 20271021: 60-observation drawdown resilience.
Higher signal means price is nearer its own trailing 60-observation high, a simple
trend-resilience measure intended to distinguish persistent leaders cross-asset.
"""
import os,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-10-20')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.close.astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index()
# log distance from trailing peak; 0 is strongest resilience, more negative is deeper drawdown.
F=np.log(P/P.rolling(60,min_periods=45).max()).loc[:END]
def metric(h):
 R=(P.shift(-h)/P-1).reindex(F.index); out=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),R.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.r).statistic
   if np.isfinite(q):out.append((dt,float(q)));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);x.index=pd.to_datetime(x.index); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments_per_ic_date':float(np.mean(ns))}
M={}
for h in [1,5,10,20]:
 x,M[h]=metric(h); print('HORIZON',h,json.dumps(M[h],sort_keys=True))
x,_=metric(5)
for lab,mask in [('2020',x.index.year==2020),('2021_2022',x.index.year.isin([2021,2022])),('2023_2024',x.index.year.isin([2023,2024])),('2025_2027',x.index.year>=2025)]:
 y=x[mask]; print('REGIME_5D',lab,'dates',len(y),'IC',float(y.mean()) if len(y) else None,'ICIR',float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,'hit',float((y>0).mean()) if len(y) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
paths={
'miner_1_volnorm_reversal_5obs':'scripts/miner_1_20260716_volnorm_reversal5_signal.pkl','miner_1_downside_upside_volatility_balance_20d':'scripts/miner_1_20261203_downside_upside_volatility_balance_20d_signal.pkl','miner_1_semivolatility_balance_improvement_10d':'scripts/miner_1_20261217_semivolatility_balance_improvement_10d_signal.pkl','miner_1_inverted_downside_cross_asset_beta_40d':'scripts/miner_1_20270114_inverted_downside_cross_asset_beta_40d_signal.pkl','miner_1_inverse_directional_recovery_efficiency_10d':'scripts/miner_1_20270603_inverse_directional_recovery_efficiency_10d_signal.pkl','miner_1_directional_volume_imbalance_30obs':'scripts/miner_1_20270909_directional_volume_imbalance_30obs_signal.pkl','miner_2_realized_volatility_20obs':'scripts/miner_2_20260716_realized_volatility20_signal.pkl','miner_2_volume_confirmed_drawdown_recovery_60d':'scripts/miner_2_20261105_volume_confirmed_drawdown_recovery_60d_signal.pkl','miner_2_inverse_return_serial_dependence_20obs':'scripts/miner_2_20270701_inverse_return_serial_dependence_20obs_signal.pkl','miner_3_relative_volume_participation_20d':'scripts/miner_3_20260716_relative_volume_participation_20d_signal.pkl','miner_3_risk_adjusted_trend_20d':'scripts/miner_3_20260716_risk_adjusted_trend_20d_signal.pkl','miner_3_vix_shock_resilience_20d':'scripts/miner_3_20260827_vix_shock_resilience_20d_signal.pkl','miner_3_drawdown_velocity_reversal_60d':'scripts/miner_3_20270408_drawdown_velocity_reversal_60d_signal.pkl','miner_3_post_recovery_reversal_20d':'scripts/miner_3_20270715_post_recovery_reversal_20d_signal.pkl'}
E={};mx=0;ok=True
for name,path in paths.items():
 if not os.path.exists(path): E[name]={'rho':None,'common_signal_cells':0};ok=False;continue
 L=pd.read_pickle(path); L.index=pd.to_datetime(L.index); L=L.reindex(index=F.index,columns=A)
 z=pd.concat([F.stack().rename('a'),L.stack().rename('b')],axis=1).dropna(); q=float(spearmanr(z.a,z.b).statistic) if len(z)>=8 else None
 E[name]={'rho':q,'common_signal_cells':len(z)};ok &=q is not None
 if q is not None:mx=max(mx,abs(q))
 print('LIBRARY_CORR',name,'cells',len(z),'spearman',q)
print('FACTOR drawdown_resilience_60obs');print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in M.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx if ok else None,'COMPLETE_EVIDENCE',ok,'EVIDENCE',json.dumps(E,sort_keys=True));F.to_pickle('scripts/miner_2_20271021_drawdown_resilience_60obs_signal.pkl')
