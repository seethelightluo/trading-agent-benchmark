"""One idea: inverse downside-regime peer-relative lag-5 serial dependence (60 sessions)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={};V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date');C[a]=pd.to_numeric(d.close,errors='coerce');V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C); V=pd.DataFrame(V).reindex(P.index);r=P.pct_change();m=r.median(axis=1);rel=r.sub(m,axis=0);cutoff=P.dropna(how='all').index.max()
def cs(x):return x.sub(x.median(axis=1),axis=0)
# Candidate: negated lag-5 serial correlation only when the cross-asset median daily return is negative
# at both observations.  Higher score is lower persistence / greater reversal after broad down sessions.
down=m<0
raw=-pd.DataFrame({a:rel[a].where(down).rolling(60,min_periods=20).corr(rel[a].shift(5).where(down.shift(5))) for a in A})
cand=cs(raw).shift(1)
# Exact operational reconstructions for all current admitted library entries, for required novelty audit.
vol20=r.rolling(20,min_periods=15).std();vol40=r.rolling(40,min_periods=15).std();trend=P.pct_change(20)/vol20
other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A});corr20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(other[a]) for a in A})
def beta(x,y,w,side=None):
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1)
 if side=='down':z=z.where(z.y<0)
 if side=='up':z=z.where(z.y>0)
 return z.x.rolling(w,min_periods=max(8,w//4)).cov(z.y)/z.y.rolling(w,min_periods=max(8,w//4)).var()
def ix(s):
 d=get_index_daily_data(s,5000).copy();d.date=pd.to_datetime(d.date);return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce').reindex(P.index)
vix=ix('VIX');dxy=ix('DXY');vr=vix.pct_change();dr=dxy.pct_change();y10=P.US10Y.pct_change();jpy=ix('USDJPY').pct_change()
neg=r.clip(upper=0);short=np.sqrt((neg*neg).rolling(10,min_periods=7).mean());long=np.sqrt((neg*neg).rolling(40,min_periods=25).mean());liq=np.log(V/V.rolling(20,min_periods=15).mean());pos=r.clip(lower=0)
q25=m.rolling(60,min_periods=40).quantile(.25);event=m<m.rolling(60,min_periods=40).quantile(.35);disp=r.std(axis=1);de=(disp>disp.rolling(60,min_periods=40).quantile(.75))&(disp>disp.shift(5));common=corr20.median(axis=1);ce=(common>common.rolling(60,min_periods=40).quantile(.75))&(common>common.shift(5))
def evmean(x,e,w,mi):return x.where(e,axis=0).rolling(w,min_periods=mi).mean()
def eb(x,y,e):
 z=pd.concat([x.rename('x'),y.rename('y'),e.rename('e')],axis=1).where(lambda q:q.e);return z.x.rolling(60,min_periods=12).cov(z.y)/z.y.rolling(60,min_periods=12).var()
peak=P.rolling(60,min_periods=45).max();dd=P/peak-1
up=pd.DataFrame({a:r[a].where(other[a]>0).rolling(60,min_periods=20).mean()-r[a].where(other[a]<0).rolling(60,min_periods=20).mean() for a in A})
dp=pd.DataFrame({a:(r[a]<0).rolling(60,min_periods=40).mean()-((r[a]<0)&(other[a]<0)).rolling(60,min_periods=40).sum()/(other[a]<0).rolling(60,min_periods=40).sum() for a in A})
S={'peer_relative_downside_volatility_compression_10_40':cs(-np.log((short+1e-5)/(long+1e-5))),'gradual_volatility_contraction_gated_trend_20':trend*np.tanh((-np.log(vol20/vol40)).clip(-2,2)),'relative_volume_participation_20d':liq,'quiet_trend_path_efficiency_20_60':P.pct_change(20).abs()/r.abs().rolling(20,min_periods=15).sum()*(1-vol20.rolling(60,min_periods=40).rank(pct=True)),'upside_return_concentration_60':pos.rolling(60,min_periods=40).max()/pos.rolling(60,min_periods=40).sum(),'inverse_idiosyncratic_volatility_20':-rel.rolling(20,min_periods=15).std(),'risk_adjusted_trend_20d':trend,'smooth_peer_relative_drawdown_recovery_60_10':cs((dd-dd.shift(10))/(.01-dd.shift(10))),'conditional_downside_participation_avoidance_60':cs(dp),'downside_correlation_regime_spread_20_80':cs(pd.DataFrame({a:beta(r[a],m,20,'down')-beta(r[a],m,80,'down') for a in A})),'conditional_peer_upside_participation_60':cs(up),'commonality_expansion_transition_40':cs(corr20.rolling(20,min_periods=15).mean()-corr20.shift(20).rolling(20,min_periods=15).mean()),'downside_cross_asset_beta_resilience_40':pd.DataFrame({a:beta(r[a],m,40,'down') for a in A}),'dxy_directional_return_asymmetry_60':pd.DataFrame({a:r[a].where(dr<0).rolling(60,min_periods=15).mean()-r[a].where(dr>0).rolling(60,min_periods=15).mean() for a in A}),'dispersion_shock_peer_reversal_20':cs(-evmean(rel.rolling(5,min_periods=5).sum().div(vol20),de,20,5)),'commonality_shock_peer_relative_reversal_60':cs(-evmean(P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0).div(vol20),ce,60,1)),'vix_regime_conditioned_risk_adjusted_trend_20':trend.mul(np.where(vix/vix.shift(20)-1>0,-1.,1.),axis=0),'stable_liquidity_participation_20':-liq.rolling(20,min_periods=15).std(),'inverse_lower_tail_persistence_40_60':-pd.DataFrame({a:r[a].lt(r[a].rolling(60,min_periods=40).quantile(.2).shift(1)).rolling(40,min_periods=25).mean() for a in A}),'vix_upside_shock_beta_resilience_40':pd.DataFrame({a:-beta(r[a],vr,40,'up') for a in A}),'volnorm_reversal_5obs':-P.pct_change(5)/r.rolling(5,min_periods=4).std(),'return_skewness_60':r.rolling(60,min_periods=40).skew(),'volscaled_reversal_1obs':-r/vol20,'inverse_peer_relative_serial_dependence_20':cs(-pd.DataFrame({a:rel[a].rolling(20,min_periods=16).corr(rel[a].shift(1)) for a in A})),'inverse_peer_relative_lag5_serial_dependence_40':cs(-pd.DataFrame({a:rel[a].rolling(40,min_periods=30).corr(rel[a].shift(5)) for a in A})).shift(1),'yield_shock_beta_resilience_60':cs(pd.DataFrame({a:eb(r[a],y10,y10.abs()>y10.abs().rolling(60,min_periods=40).quantile(.75))-beta(r[a],y10,60) for a in A})),'yield_volatility_transition_beta_resilience_60':cs(pd.DataFrame({a:eb(r[a],y10,y10.abs().rolling(20,min_periods=15).mean()>y10.abs().rolling(60,min_periods=40).mean())-beta(r[a],y10,60) for a in A})),'usdjpy_volatility_transition_beta_resilience_60':cs(pd.DataFrame({a:eb(r[a],jpy,jpy.abs().rolling(20,min_periods=15).mean()>jpy.abs().rolling(60,min_periods=40).mean())-beta(r[a],jpy,60) for a in A}))}
fw={h:P.shift(-h)/P-1 for h in(1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1];z=[];ns=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z)
 return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),6),'breadth':round(np.mean(ns),3),'min_breadth':min(ns)}
print('FACTOR inverse_downside_regime_peer_relative_lag5_serial_dependence_60 CUTOFF',cutoff.date(),'ASSETS',len(A));print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(cand.std(axis=1).mean(),6))
for h in(1,5,10,20):print('H',h,stat(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,p))
mx=0;who='';evidence=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 if abs(rho)>mx:mx=abs(rho);who=n;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
