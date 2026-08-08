# Miner 2 scheduled revalidation: Commonality Expansion Transition (40 sessions).
# Uses only runtime-visible daily observations; at 2028-01-27 data endpoint must be the prior completed session.
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; px={}; vv={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date); d=d.set_index('date').sort_index().sort_index()
 px[a]=pd.to_numeric(d.close,errors='coerce'); vv[a]=pd.to_numeric(d.get('volume'),errors='coerce').replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(fill_method=None); M=R.median(axis=1); V=pd.DataFrame(vv).reindex(P.index)
sd=R.rolling(20,min_periods=15).std(); trend=P.pct_change(20,fill_method=None)/sd
peer=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(R.drop(columns=a).median(axis=1)) for a in A})
# Exact persisted definition: current 20-session mean peer correlation minus preceding non-overlapping 20-session mean.
F=peer.rolling(20,min_periods=15).mean()-peer.shift(20).rolling(20,min_periods=15).mean()
down=M<0
def beta(a,mask,w=60,minp=15):
 x=R[a].where(mask); y=M.where(mask); return x.rolling(w,min_periods=minp).cov(y)/y.rolling(w,min_periods=minp).var()
ac=-R.rolling(20,min_periods=15).apply(lambda z:z.dropna().autocorr(1) if len(z.dropna())>=15 else np.nan,raw=False)
rev5=-P.pct_change(5,fill_method=None)/R.rolling(5,min_periods=4).std(); rev1=-R/sd
trans=ac*np.log(sd.rolling(5,min_periods=4).mean()/sd).clip(-2,2)
peer40=pd.DataFrame({a:R[a].rolling(40,min_periods=25).corr(R.drop(columns=a).median(axis=1)) for a in A})
eff=P.pct_change(20,fill_method=None).abs()/R.abs().rolling(20,min_periods=15).sum(); vp=sd.rolling(60,min_periods=40).rank(pct=True)
rv=np.log(V/V.rolling(20,min_periods=15).mean()); idio=-R.sub(M,axis=0).rolling(20,min_periods=15).std(); skew=R.rolling(60,min_periods=40).skew()
excess=R.sub(M,axis=0); event=-excess.where(M<M.rolling(40,min_periods=25).quantile(.25)).rolling(40,min_periods=12).mean()
tail=-R.where(R.lt(R.rolling(60,min_periods=40).quantile(.2))).rolling(40,min_periods=12).mean(); upconc=R.clip(lower=0).rolling(60,min_periods=40).max()/R.clip(lower=0).rolling(60,min_periods=40).sum()
gate=sd.rolling(20,min_periods=15).mean()<sd.rolling(40,min_periods=30).mean()
LIB={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev5,'risk_adjusted_trend_20d':trend,'downside_cross_asset_beta_resilience_40':pd.DataFrame({a:-beta(a,down,40,12) for a in A}),'inverse_idiosyncratic_volatility_20':idio,'stable_liquidity_participation_20':-rv.rolling(20,min_periods=15).std(),'return_skewness_60':skew,'gradual_volatility_contraction_gated_trend_20':trend.where(gate),'dxy_directional_return_asymmetry_60':None,'inverse_lag1_return_autocorrelation_20':ac,'volatility_transition_serial_resilience_20':trans,'low_commonality_other_median_correlation_40':-peer40,'downside_upside_cross_asset_beta_asymmetry_60':pd.DataFrame({a:beta(a,down)-beta(a,~down) for a in A}),'relative_volume_participation_20d':rv,'quiet_trend_path_efficiency_20_60':eff*(1-vp),'vix_regime_conditioned_risk_adjusted_trend_20':None,'vix_upside_shock_beta_resilience_40':None,'downside_event_excess_magnitude_median_40':event,'inverse_lower_tail_persistence_40_60':tail,'upside_return_concentration_60':upconc}
try:
 d=get_index_daily_data('VIX',5000).copy(); d['date']=pd.to_datetime(d.date); ix=pd.to_numeric(d.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill(); vr=ix.pct_change(); vu=vr>0
 LIB['vix_regime_conditioned_risk_adjusted_trend_20']=trend.where(ix.pct_change(20)<=0,-trend)
 LIB['vix_upside_shock_beta_resilience_40']=pd.DataFrame({a:-R[a].where(vu).rolling(40,min_periods=12).cov(vr.where(vu))/vr.where(vu).rolling(40,min_periods=12).var() for a in A})
 d=get_index_daily_data('DXY',5000).copy(); d['date']=pd.to_datetime(d.date); dx=pd.to_numeric(d.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill().pct_change()
 LIB['dxy_directional_return_asymmetry_60']=pd.DataFrame({a:R[a].where(dx>0).rolling(60,min_periods=15).mean()-R[a].where(dx<=0).rolling(60,min_periods=15).mean() for a in A})
except Exception as e: print('MACRO_READ_FAILURE',e)
def stats(h,lo=None,hi=None):
 fw=P.shift(-h)/P-1; ics=[]; ns=[]; dates=F.index if lo is None else F.loc[lo:hi].index
 for t in dates:
  z=pd.concat([F.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): ics.append(q);ns.append(len(z))
 if not ics:return [0,np.nan,np.nan,np.nan,np.nan,np.nan]
 s=pd.Series(ics);return [len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),np.mean(ns),min(ns)]
def fm(x):return 'NA' if not np.isfinite(x) else f'{x:.6f}'
print('FACTOR commonality_expansion_transition_40 ENDPOINT',P.index.max().date(),'PERIOD',P.index.min().date(),P.index.max().date(),'ASSETS',len(A))
print('COVERAGE',int(F.notna().sum().sum()),'OF',F.size,'RATE',fm(F.notna().mean().mean()))
for h in [1,5,10,20]:
 x=stats(h);print('H',h,'DATES',x[0],'IC',fm(x[1]),'ICIR',fm(x[2]),'HIT',fm(x[3]),'MEAN_NAMES',fm(x[4]),'MIN_NAMES',fm(x[5]))
for lab,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-12-31'),('2028YTD','2028-01-01','2028-01-26')]:
 x=stats(10,lo,hi);print('REGIME10',lab,'DATES',x[0],'IC',fm(x[1]),'ICIR',fm(x[2]),'HIT',fm(x[3]))
print('TURNOVER',fm(F.rank(axis=1,pct=True).diff().abs().stack().mean()))
mx=-1;who='';evidence=0
for n,x in LIB.items():
 if x is None: print('LIBRARY',n,'RHO','NA','CELLS',0);continue
 z=pd.concat([F.stack(),x.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>2 else np.nan
 print('LIBRARY',n,'RHO',fm(q),'CELLS',len(z));evidence+=int(np.isfinite(q))
 if np.isfinite(q) and abs(q)>mx:mx=abs(q);who=n
print('LIBRARY_EVIDENCE',evidence,'OF',len(LIB));print('MAX_ABS_LIBRARY_CORRELATION',fm(mx),'FACTOR',who)
