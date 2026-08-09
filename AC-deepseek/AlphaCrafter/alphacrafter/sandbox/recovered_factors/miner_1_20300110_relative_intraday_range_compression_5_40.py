"""One idea: relative intraday-range compression, measuring short/long normalized range contraction."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={};H={};L={};V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce'); H[a]=pd.to_numeric(d.high,errors='coerce'); L[a]=pd.to_numeric(d.low,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C); hi=pd.DataFrame(H).reindex(P.index); lo=pd.DataFrame(L).reindex(P.index); V=pd.DataFrame(V).reindex(P.index); r=P.pct_change(); m=r.median(axis=1); rel=r.sub(m,axis=0); cutoff=P.dropna(how='all').index.max()
def cs(x): return x.sub(x.median(axis=1),axis=0)
def beta(x,y,w=40,which=None):
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1)
 if which=='down':z=z.where(z.y<0)
 if which=='up':z=z.where(z.y>0)
 return z.x.rolling(w,min_periods=max(8,w//4)).cov(z.y)/z.y.rolling(w,min_periods=max(8,w//4)).var()
def eventmean(x,e,w,minn=1): return x.where(e,axis=0).rolling(w,min_periods=minn).mean()
vol20=r.rolling(20,min_periods=15).std(); trend=P.pct_change(20)/vol20
other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A}); corr20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(other[a]) for a in A})
def ix(sym):
 d=get_index_daily_data(sym,5000).copy();d.date=pd.to_datetime(d.date);return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce').reindex(P.index)
vix=ix('VIX');dxy=ix('DXY');vr=vix.pct_change();dr=dxy.pct_change();y10=r['US10Y']
def eventbeta(x,y,e,w=60):
 z=pd.concat([x.rename('x'),y.rename('y'),e.rename('e')],axis=1);z[['x','y']]=z[['x','y']].where(z.e,axis=0)
 return z.x.rolling(w,min_periods=15).cov(z.y)/z.y.rolling(w,min_periods=15).var()
# Candidate: short-run normalized high-low range divided by its structural range, inverted then cross-sectionally centered.
# A low ratio identifies assets that have recently compressed their intraday trading range relative to their own baseline.
rng=(hi-lo).abs()/P.shift(1).abs(); cand=cs(-np.log((rng.rolling(5,min_periods=4).mean()+1e-7)/(rng.rolling(40,min_periods=25).mean()+1e-7))).shift(1)
peak=P.rolling(60,min_periods=45).max();dd=P/peak-1; neg=r.clip(upper=0);short=np.sqrt((neg*neg).rolling(10,min_periods=7).mean());long=np.sqrt((neg*neg).rolling(40,min_periods=25).mean());market_event=m<m.rolling(60,min_periods=40).quantile(.35);disp=r.std(axis=1);de=(disp>disp.rolling(60,min_periods=40).quantile(.75))&(disp>disp.shift(5));common=corr20.median(axis=1);ce=(common>common.rolling(60,min_periods=40).quantile(.75))&(common>common.shift(5));q25=m.rolling(60,min_periods=40).quantile(.25);vol5=r.rolling(5,min_periods=4).std();vol40=r.rolling(40,min_periods=15).std();liq=np.log(V/V.rolling(20,min_periods=15).mean());pos=r.clip(lower=0)
up=pd.DataFrame({a:r[a].where(other[a]>0).rolling(60,min_periods=20).mean()-r[a].where(other[a]<0).rolling(60,min_periods=20).mean() for a in A});downpart=pd.DataFrame({a:(r[a]<0).rolling(60,min_periods=40).mean()-((r[a]<0)&(other[a]<0)).rolling(60,min_periods=40).sum()/(other[a]<0).rolling(60,min_periods=40).sum() for a in A})
S={'peer_relative_downside_volatility_compression_10_40':cs(-np.log((short+1e-5)/(long+1e-5))),'gradual_volatility_contraction_gated_trend_20':trend*np.tanh((-np.log(vol20/vol40)).clip(-2,2)),'relative_volume_participation_20d':liq,'quiet_trend_path_efficiency_20_60':P.pct_change(20).abs()/r.abs().rolling(20,min_periods=15).sum()*(1-vol20.rolling(60,min_periods=40).rank(pct=True)),'inverse_volume_weighted_peer_tail_asymmetry_60':cs(-((rel.clip(lower=0)*liq.rolling(60,min_periods=30).rank(pct=True)).rolling(60,min_periods=40).sum()-((-rel).clip(lower=0)*liq.rolling(60,min_periods=30).rank(pct=True)).rolling(60,min_periods=40).sum())/((rel.abs()*liq.rolling(60,min_periods=30).rank(pct=True)).rolling(60,min_periods=40).sum())),'upside_return_concentration_60':pos.rolling(60,min_periods=40).max()/pos.rolling(60,min_periods=40).sum(),'inverse_idiosyncratic_volatility_20':-rel.rolling(20,min_periods=15).std(),'risk_adjusted_trend_20d':trend,'smooth_peer_relative_drawdown_recovery_60_10':cs((dd-dd.shift(10))/(.01-dd.shift(10))),'downside_event_excess_magnitude_median_40':cs(rel.where(market_event.shift(1),axis=0).rolling(40,min_periods=12).median()),'conditional_downside_participation_avoidance_60':cs(downpart),'downside_correlation_regime_spread_20_80':cs(pd.DataFrame({a:beta(r[a],m,20,'down')-beta(r[a],m,80,'down') for a in A})),'conditional_peer_upside_participation_60':cs(up),'ravmom_20obs':trend,'post_stress_relative_rebound_reversal_60':cs(-eventmean(rel.div(rel.abs().median(axis=1),axis=0),m.shift(1)<q25.shift(1),60,12)),'commonality_expansion_transition_40':cs(corr20.rolling(20,min_periods=15).mean()-corr20.shift(20).rolling(20,min_periods=15).mean()),'downside_cross_asset_beta_resilience_40':pd.DataFrame({a:beta(r[a],m,40,'down') for a in A}),'dxy_directional_return_asymmetry_60':pd.DataFrame({a:r[a].where(dr<0).rolling(60,min_periods=15).mean()-r[a].where(dr>0).rolling(60,min_periods=15).mean() for a in A}),'yield_shock_beta_resilience_60':cs(pd.DataFrame({a:eventbeta(r[a],y10,(y10.abs()>y10.abs().rolling(60,min_periods=40).quantile(.75)),60) for a in A})-pd.DataFrame({a:beta(r[a],y10,60) for a in A})),'dispersion_shock_peer_reversal_20':cs(-eventmean(rel.rolling(5,min_periods=5).sum().div(vol20),de,20,5)),'commonality_shock_peer_relative_reversal_60':cs(-eventmean((P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0)).div(vol20),ce,60,1)),'vix_regime_conditioned_risk_adjusted_trend_20':trend.mul(np.where(vix/vix.shift(20)-1>0,-1.,1.),axis=0),'stable_liquidity_participation_20':-liq.rolling(20,min_periods=15).std(),'inverse_lower_tail_persistence_40_60':-pd.DataFrame({a:r[a].lt(r[a].rolling(60,min_periods=40).quantile(.2).shift(1)).rolling(40,min_periods=25).mean() for a in A}),'vix_upside_shock_beta_resilience_40':pd.DataFrame({a:-beta(r[a],vr,40,'up') for a in A}),'volnorm_reversal_5obs':-P.pct_change(5)/vol5,'delayed_post_stress_relative_rebound_reversal_60':cs(-eventmean(P.pct_change(3).sub(P.pct_change(3).median(axis=1),axis=0).div(P.pct_change(3).sub(P.pct_change(3).median(axis=1),axis=0).abs().median(axis=1),axis=0),m.shift(5)<q25.shift(5),60,12)),'return_skewness_60':r.rolling(60,min_periods=40).skew(),'volscaled_reversal_1obs':-r/vol20}
print('FACTOR relative_intraday_range_compression_5_40 CUTOFF',cutoff.date(),'ASSETS',len(A));fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];y=fw[h].reindex(x.index);z=[];n=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);n.append(len(q))
 if not z:return {}
 z=np.array(z);return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),6),'breadth':round(np.mean(n),3),'min_breadth':min(n)}
print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(cand.std(axis=1).mean(),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=0;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=3 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1 else np.nan
 if abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>2 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6))
