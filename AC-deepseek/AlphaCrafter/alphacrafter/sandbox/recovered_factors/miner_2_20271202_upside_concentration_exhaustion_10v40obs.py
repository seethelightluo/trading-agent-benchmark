"""miner_2 20271202: upside-concentration exhaustion, strict admitted-library test."""
import os,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-01')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:END];r=P.pct_change()
# Negative orientation is specified ex ante: concentrated recent upside is an exhaustion signal, so lower concentration ranks higher.
F=-(r.clip(lower=0).rolling(10,min_periods=8).sum()/r.clip(lower=0).rolling(40,min_periods=28).sum())
def met(h):
 out=[];nn=[]; R=P.shift(-h)/P-1
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append((d,q));nn.append(len(z))
 x=pd.Series(dict(out));s=x.std(ddof=1);return x,dict(daily_paper_ic=float(x.mean()),daily_paper_icir=float(x.mean()/s),ic_hit_ratio=float((x>0).mean()),ic_standard_error=float(s/np.sqrt(len(x))),ic_dates=len(x),mean_valid_instruments_per_ic_date=float(np.mean(nn)))
M={}
for h in [1,5,10,20]:x,M[h]=met(h);print('HORIZON',h,json.dumps(M[h]))
x,_=met(5)
for lab,mask in [('2020',x.index.year==2020),('2021_2022',x.index.year.isin([2021,2022])),('2023_2024',x.index.year.isin([2023,2024])),('2025_2027',x.index.year>=2025)]:
 y=x[mask];print('REGIME_5D',lab,len(y),float(y.mean()) if len(y) else None,float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,float((y>0).mean()) if len(y) else None)
paths={'miner_1_volnorm_reversal_5obs':'scripts/miner_1_20260716_volnorm_reversal5_signal.pkl','miner_1_downside_upside_volatility_balance_20d':'scripts/miner_1_20261203_downside_upside_volatility_balance_20d_signal.pkl','miner_1_semivolatility_balance_improvement_10d':'scripts/miner_1_20261217_semivolatility_balance_improvement_10d_signal.pkl','miner_1_inverted_downside_cross_asset_beta_40d':'scripts/miner_1_20270114_inverted_downside_cross_asset_beta_40d_signal.pkl','miner_1_inverse_directional_recovery_efficiency_10d':'scripts/miner_1_20270603_inverse_directional_recovery_efficiency_10d_signal.pkl','miner_1_directional_volume_imbalance_30obs':'scripts/miner_1_20270909_directional_volume_imbalance_30obs_signal.pkl','miner_2_realized_volatility_20obs':'scripts/miner_2_20260716_realized_volatility20_signal.pkl','miner_2_volume_confirmed_drawdown_recovery_60d':'scripts/miner_2_20261105_volume_confirmed_drawdown_recovery_60d_signal.pkl','miner_2_inverse_return_serial_dependence_20obs':'scripts/miner_2_20270701_inverse_return_serial_dependence_20obs_signal.pkl','miner_2_downside_concentration_continuation_10v40obs':'scripts/miner_2_20271118_downside_concentration_continuation_10v40obs_signal.pkl','miner_3_relative_volume_participation_20d':'scripts/miner_3_20260716_relative_volume_participation_20d_signal.pkl','miner_3_risk_adjusted_trend_20d':'scripts/miner_3_20260716_risk_adjusted_trend_20d_signal.pkl','miner_3_vix_shock_resilience_20d':'scripts/miner_3_20260827_vix_shock_resilience_20d_signal.pkl','miner_3_drawdown_velocity_reversal_60d':'scripts/miner_3_20270408_drawdown_velocity_reversal_60d_signal.pkl','miner_3_post_recovery_reversal_20d':'scripts/miner_3_20270715_post_recovery_reversal_20d_signal.pkl'}
mx=0;who=None;ok=True
for n,p in paths.items():
 if not os.path.exists(p):ok=False;print('MISSING',n);continue
 L=pd.read_pickle(p);L.index=pd.to_datetime(L.index);z=pd.concat([F.stack(),L.reindex(index=F.index,columns=A).stack()],axis=1).dropna();q=float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic) if len(z)>=8 else None
 print('LIBRARY_CORR',n,len(z),q);ok &=q is not None
 if q is not None and abs(q)>mx:mx=abs(q);who=n
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('SUMMARY',json.dumps({'period':f'{F.index.min().date()} to {END.date()}','panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'max_abs_library_correlation':mx if ok else None,'most_correlated':who,'complete':ok,'decay':M}))
F.to_pickle('scripts/miner_2_20271202_upside_concentration_exhaustion_10v40obs_signal.pkl')
