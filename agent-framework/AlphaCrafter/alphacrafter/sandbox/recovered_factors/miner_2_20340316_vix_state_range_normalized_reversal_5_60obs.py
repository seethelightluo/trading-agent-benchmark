"""Miner 2: VIX-state conditioned, range-normalized 5-day reversal, 60-session state normalization."""
import pandas as pd, numpy as np
old=open('scripts/miner_2_20340202_lagged_common_stress_range_expansion_sensitivity_60obs.py').read()
head=old[:old.index("# Price-only liquidity")].replace("END=pd.Timestamp('2034-02-01')", "END=pd.Timestamp('2034-03-15')")
exec(head)
# A short-term price reversal standardized by own completed-bar intraday range,
# admitted only when lagged VIX is above its own trailing median. This separates
# liquidity-driven overshoots during macro-volatility from unconditional reversal.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).reindex(p.index).ffill()
vix_high=vix.shift(1).gt(vix.rolling(60,min_periods=45).median().shift(1))
rng=np.log(hi.clip(lower=1e-12).div(lo.clip(lower=1e-12)))
scale=rng.rolling(20,min_periods=15).mean().replace(0,np.nan)
f=(-(p.div(p.shift(5))-1).div(scale)).where(vix_high,axis=0)
print('CANDIDATE vix_state_conditioned_range_normalized_reversal_5_60obs endpoint',p.index.max().date(),'assets',len(A),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
exec(old[old.index('R={}'):old.index('audit=src[')])
audit=src[src.index('# Full current-library reconstruction'):];exec(audit[:audit.index('mx=-1')])
L['normal_dispersion_residual_downside_upside_magnitude_asymmetry_60obs']=asym
L['normal_dispersion_continuous_downside_residual_rebound_efficiency_60obs']=base
nrk=res.rank(axis=1,pct=True);nd=(nrk.shift(2)-.5).abs();nmv=(nd-(nrk.shift(1)-.5).abs()).where(nd.ge(.30)).where(~high,axis=0);nn=nmv.notna().astype(float).rolling(60,min_periods=20).sum();L['normal_dispersion_residual_extreme_rank_normalization_60obs']=nmv.rolling(60,min_periods=20).mean().where(nn>=20)
aabs=asym.abs();stable=aabs.le(aabs.rolling(60,min_periods=45).median())
L['own_history_stable_asymmetry_normal_dispersion_downside_rebound_efficiency_60obs']=base.where(stable)
L['continuous_asymmetry_penalized_normal_dispersion_downside_rebound_efficiency_60obs']=base*(1-aabs.clip(0,1))
mx=-1;missing=[]
for n,vv in L.items():
 q=pd.concat([f.stack(),vv.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)<8: missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)
# Explicit selected-horizon regime and turnover report.
for h,rr in R.items():
 print('REGIMES horizon',h)
 for label,lo_,up in [('2026_2029','2026-01-01','2029-12-31'),('2030_2032','2030-01-01','2032-12-31'),('recent_2033_2034','2033-01-01','2034-03-15')]:
  vals,dates,_=rr; z=np.asarray(vals)[(dates>=pd.Timestamp(lo_))&(dates<=pd.Timestamp(up))]
  print(label,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/(z.std(ddof=1)+1e-12),6) if len(z)>1 else None,'hit',round((z>0).mean(),6) if len(z) else None)
rk=f.rank(axis=1); ts=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: ts.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('TURNOVER valid_pairs',len(ts),'mean_rank_turnover',round(np.mean(ts),6) if ts else None,'median_IQR',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
