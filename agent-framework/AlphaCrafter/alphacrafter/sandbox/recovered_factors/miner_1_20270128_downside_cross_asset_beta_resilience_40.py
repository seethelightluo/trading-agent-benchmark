"""miner_1: downside cross-asset beta resilience, one interpretable candidate."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2027-01-27'); H=[1,5,10,20]
C={}; R={}; V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date')
 C[a]=d.close.replace(0,np.nan); R[a]=C[a].pct_change(); V[a]=d.volume.replace(0,np.nan)
r=pd.DataFrame(R).sort_index(); market=r.median(axis=1,skipna=True)
# Factor is negative beta to the cross-asset median on negative-median days: resilience/high score means lower shared downside exposure.
def downside_beta(x):
 x=pd.Series(x); m=market.reindex(x.index); ok=(m<0)&x.notna()&m.notna()
 if ok.sum()<12 or m[ok].var(ddof=1)==0:return np.nan
 return -np.cov(x[ok],m[ok],ddof=1)[0,1]/m[ok].var(ddof=1)
factor=pd.DataFrame({a:R[a].rolling(40,min_periods=25).apply(lambda x: downside_beta(x),raw=False) for a in A})
fwd={h:pd.DataFrame({a:C[a].shift(-h)/C[a]-1 for a in A}) for h in H}
def ev(h,span=None):
 f=factor if span is None else factor.loc[span[0]:span[1]]; y=fwd[h] if span is None else fwd[h].loc[span[0]:span[1]]; z=[];ns=[];const=0
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
   else:const+=1
 z=np.asarray(z)
 return {'dates':len(z),'dropped_constant':const,'ic':float(z.mean()),'icir':float(z.mean()/z.std(ddof=1)),'hit':float((z>0).mean()),'mean_names':float(np.mean(ns)),'min_names':int(np.min(ns))} if len(z) else {'dates':0}
print('FACTOR downside_cross_asset_beta_resilience_40 cutoff',cutoff.date())
print('CELLS',int(factor.notna().sum().sum()),'/',factor.size,'coverage',float(factor.notna().stack().mean()))
for h in H:print('H',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_ytd',('2027-01-01','2027-01-27'))]:print('REGIME_10',n,ev(10,s))
print('TURNOVER',float(factor.rank(axis=1,pct=True).diff().abs().stack().mean()))
# Reconstruct every currently admitted signal for mandatory complete correlation evidence.
def quiet(a):return (C[a].pct_change(20).abs()/R[a].abs().rolling(20,min_periods=15).sum())*(1-R[a].rolling(20,min_periods=15).std().rolling(60,min_periods=40).apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1],raw=False))
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.replace(0,np.nan).reindex(r.index).ffill();vr=vix/vix.shift(20)-1
# upside VIX beta conditional observation window
def vixbeta(x):
 x=pd.Series(x); vv=vix.reindex(x.index).pct_change(); ok=(vv>0)&x.notna()&vv.notna()
 if ok.sum()<10 or vv[ok].var(ddof=1)==0:return np.nan
 return -np.cov(x[ok],vv[ok],ddof=1)[0,1]/vv[ok].var(ddof=1)
def invac(x):
 x=pd.Series(x).dropna()
 return np.nan if len(x)<15 else -x.autocorr(lag=1)
lib={
'relative_volume_participation':pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}),
'risk_adjusted_trend':pd.DataFrame({a:(C[a]/C[a].shift(20)-1)/R[a].rolling(20,min_periods=15).std() for a in A}),
'ravmom_20obs':pd.DataFrame({a:(C[a]/C[a].shift(20)-1)/R[a].rolling(20,min_periods=15).std() for a in A}),
'volnorm_reversal_5':pd.DataFrame({a:-(C[a]/C[a].shift(5)-1)/R[a].rolling(5,min_periods=4).std() for a in A}),
'volscaled_reversal_1':pd.DataFrame({a:-R[a]/R[a].rolling(20,min_periods=15).std() for a in A}),
'quiet_trend_path_efficiency':pd.DataFrame({a:quiet(a) for a in A}),
'inverse_lag1_return_autocorrelation_20':pd.DataFrame({a:R[a].rolling(20,min_periods=15).apply(invac,raw=False) for a in A}),
'vix_regime_conditioned_risk_adjusted_trend':pd.DataFrame({a:np.where(vr>0,-1,1)*((C[a]/C[a].shift(20)-1)/R[a].rolling(20,min_periods=15).std()) for a in A}),
'vix_upside_shock_beta_resilience_40':pd.DataFrame({a:R[a].rolling(40,min_periods=25).apply(vixbeta,raw=False) for a in A})}
mx=0
for n,x in lib.items():
 q=pd.concat([factor.stack(),x.stack()],axis=1).dropna(); rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);mx=max(mx,abs(rho));print('LIBCORR',n,'cells',len(q),'rho',rho)
print('MAX_ABS_LIBRARY_CORRELATION',mx)
