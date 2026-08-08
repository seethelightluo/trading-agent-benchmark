"""Miner 2: residual VIX-change sensitivity, 60 sessions."""
import pandas as pd, numpy as np
old=open('scripts/miner_2_20331222_vix_normalized_lagged_common_shock_response_60obs.py').read()
head=old[:old.index("# Asset-specific")].replace("END=pd.Timestamp('2033-12-21')", "END=pd.Timestamp('2034-01-04')")
exec(head)
# Cross-sectionally varying defensive transmission: each asset's rolling
# idiosyncratic-return sensitivity to prior completed-session VIX changes.
# VIX itself is observation-only; beta is estimated exclusively from lagged bars.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).reindex(p.index)
dv=np.log(vix).diff().shift(1)
def beta_to(x,y):
 return x.rolling(60,min_periods=45).cov(y).div(y.rolling(60,min_periods=45).var()+1e-12)
f=pd.DataFrame({a:beta_to(res[a],dv) for a in A})
print('CANDIDATE residual_vix_change_sensitivity_60obs endpoint',p.index.max().date(),'assets',len(A),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
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
