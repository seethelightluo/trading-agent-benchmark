"""Miner 1 -- cross-asset downside recovery consistency (60d)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; px={};vol={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 px[a]=pd.to_numeric(d.close,errors='coerce'); vol[a]=pd.to_numeric(d.get('volume'),errors='coerce').replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(fill_method=None); V=pd.DataFrame(vol).reindex(P.index); M=R.median(axis=1); sd=R.rolling(20,min_periods=15).std()
# A market lower-tail event is known at close t-1. Score whether each asset has consistently beaten
# the cross-sectional median on the following session, measured over past 60 fully completed observations.
cut=M.rolling(60,min_periods=40).quantile(.30).shift(1); event=M.shift(1).lt(cut.shift(1))
ex=R.sub(M,axis=0); F=ex.where(event,axis=0).gt(0).rolling(60,min_periods=12).mean().shift(1)
trend=P.pct_change(20,fill_method=None)/sd; rev5=-P.pct_change(5,fill_method=None)/R.rolling(5,min_periods=4).std(); rev1=-R/sd
inva=-R.rolling(20,min_periods=15).apply(lambda z:z.dropna().autocorr(1) if len(z.dropna())>=15 else np.nan,raw=False)
trans=inva*np.log(sd.rolling(5,min_periods=4).mean()/sd).clip(-2,2); oth=pd.DataFrame({a:R[a].rolling(40,min_periods=25).corr(R.drop(columns=a).median(axis=1)) for a in A})
eff=P.pct_change(20,fill_method=None).abs()/R.abs().rolling(20,min_periods=15).sum(); vp=sd.rolling(60,min_periods=40).rank(pct=True)
corr=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(R.drop(columns=a).median(axis=1)) for a in A}); exp=corr.rolling(20,min_periods=15).mean()-corr.shift(20).rolling(20,min_periods=15).mean()
down=M<0; betaD=pd.DataFrame({a:R[a].where(down).rolling(60,min_periods=12).cov(M.where(down))/M.where(down).rolling(60,min_periods=12).var()-R[a].where(~down).rolling(60,min_periods=12).cov(M.where(~down))/M.where(~down).rolling(60,min_periods=12).var() for a in A})
lowtail=-R.rolling(60,min_periods=40).quantile(.15).abs()/sd; gate=sd.rolling(20,min_periods=15).mean()<sd.rolling(40,min_periods=30).mean()
# Exact admitted factor reconstructions.
LIB={'gradual_volatility_contraction_gated_trend_20':trend.where(gate),'downside_upside_cross_asset_beta_asymmetry_60':betaD,'miner_3_relative_volume_participation_20d':np.log(V/V.rolling(20,min_periods=15).mean()),'miner_3_quiet_trend_path_efficiency_20_60':eff*(1-vp),'miner_1_inverse_idiosyncratic_volatility_20':-R.sub(M,axis=0).rolling(20,min_periods=15).std(),'miner_3_risk_adjusted_trend_20d':trend,'miner_3_downside_event_excess_magnitude_median_40':R.sub(M,axis=0).where((M<M.rolling(60,min_periods=40).quantile(.35)).shift(1),axis=0).rolling(40,min_periods=12).median(),'low_commonality_other_median_correlation_40':-oth,'miner_1_ravmom_20obs':trend,'commonality_expansion_transition_40':exp,'miner_1_downside_cross_asset_beta_resilience_40':-pd.DataFrame({a:R[a].where(down).rolling(40,min_periods=12).cov(M.where(down))/M.where(down).rolling(40,min_periods=12).var() for a in A}),'miner_2_inverse_lag1_return_autocorrelation_20':inva,'volatility_transition_serial_resilience_20':trans,'stable_liquidity_participation_20':-np.log(V/V.rolling(20,min_periods=15).mean()).rolling(20,min_periods=15).std(),'inverse_lower_tail_persistence_40_60':lowtail,'miner_1_volnorm_reversal_5obs':rev5,'return_skewness_60':R.rolling(60,min_periods=40).skew(),'miner_2_volscaled_reversal_1obs':rev1}
ix=get_index_daily_data('VIX',5000).copy();ix.date=pd.to_datetime(ix.date); vx=pd.to_numeric(ix.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill(); vr=vx.pct_change();up=vr>0
LIB['vix_regime_conditioned_risk_adjusted_trend_20']=trend.where(vx.pct_change(20)<=0,-trend);LIB['miner_3_vix_upside_shock_beta_resilience_40']=pd.DataFrame({a:-R[a].where(up).rolling(40,min_periods=12).cov(vr.where(up))/vr.where(up).rolling(40,min_periods=12).var() for a in A})
def stats(h,lo=None,hi=None):
 out=[]; nn=[]; fw=P.shift(-h)/P-1
 for t in F.loc[lo:hi].index if lo else F.index:
  z=pd.concat([F.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append(q);nn.append(len(z))
 s=pd.Series(out); return len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),np.mean(nn),min(nn)
def fmt(x):return 'NA' if not np.isfinite(x) else f'{x:.6f}'
print('FACTOR cross_asset_downside_recovery_consistency_60 ENDPOINT',P.index.max().date(),'PERIOD',P.index.min().date(),P.index.max().date(),'ASSETS',len(A)); print('COVERAGE',F.notna().sum().sum(),'OF',F.size,'RATE',fmt(F.notna().mean().mean()))
for h in [1,5,10,20]:
 x=stats(h);print('H',h,'DATES',x[0],'IC',fmt(x[1]),'ICIR',fmt(x[2]),'HIT',fmt(x[3]),'MEAN_NAMES',fmt(x[4]),'MIN_NAMES',fmt(x[5]))
for n,l,u in [('2020','2020-01-01','2020-12-31'),('2021_22','2021-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027YTD','2027-01-01','2027-12-31')]:
 x=stats(10,l,u);print('REGIME10',n,'DATES',x[0],'IC',fmt(x[1]),'ICIR',fmt(x[2]),'HIT',fmt(x[3]))
print('TURNOVER',fmt(F.rank(axis=1,pct=True).diff().abs().stack().mean()))
mx=-1; who=''
for n,x in LIB.items():
 z=pd.concat([F.stack(),x.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
 print('LIBRARY',n,'RHO',fmt(q),'CELLS',len(z))
 if np.isfinite(q) and abs(q)>mx:mx=abs(q);who=n
print('MAX_ABS_LIBRARY_CORRELATION',fmt(mx),'FACTOR',who)
