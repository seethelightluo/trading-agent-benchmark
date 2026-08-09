"""Miner 2: DXY-shock-conditioned relative-volume response, 60 sessions."""
import pandas as pd, numpy as np
old=open('scripts/miner_2_20340105_residual_vix_change_sensitivity_60obs.py').read()
head=old[:old.index("# Cross-sectionally varying")].replace("END=pd.Timestamp('2034-01-04')", "END=pd.Timestamp('2034-01-18')")
exec(head)
# Cross-asset liquidity transmission: an asset scores highly when its abnormal
# own trading volume rises systematically following unusually large *lagged* DXY
# moves. DXY is observation-only. The signal uses a rolling correlation, so it
# measures a distinct response profile rather than a common macro state.
dxy=macro('DXY').reindex(p.index)
dxy_shock=dxy.abs().shift(1)
relvol=np.log(vo.div(vo.rolling(20,min_periods=15).mean()))
def corr_to(x,y):
 return x.rolling(60,min_periods=45).cov(y).div(x.rolling(60,min_periods=45).std()*y.rolling(60,min_periods=45).std()+1e-12)
f=pd.DataFrame({a:corr_to(relvol[a],dxy_shock) for a in A})
print('CANDIDATE dxy_shock_conditioned_relative_volume_response_60obs endpoint',p.index.max().date(),'assets',len(A),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
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
