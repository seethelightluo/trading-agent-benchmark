"""miner_1: VIX-regime-conditioned risk-adjusted 20-observation trend.
The sign of the trend score is inverted when VIX has risen over 20 observations.
All inputs are truncated at cutoff; VIX is observation-only."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2026-12-02'); H=[1,5,10,20]
C={}; R={}; V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date')
 C[a]=d.close.replace(0,np.nan); R[a]=C[a].pct_change(); V[a]=d.volume.replace(0,np.nan)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.replace(0,np.nan)
# Only completed contemporaneous VIX values: reindexing never draws from after each asset date.
vix_change=vix.pct_change(20)
def regime(ix):
 x=vix_change.reindex(ix,method='ffill')
 return pd.Series(np.where(x>0,-1.,1.),index=ix).where(x.notna())
base=pd.DataFrame({a:(C[a]/C[a].shift(20)-1)/R[a].rolling(20,min_periods=15).std() for a in A})
factor=base.mul(regime(base.index),axis=0)
fwd={h:pd.DataFrame({a:C[a].shift(-h)/C[a]-1 for a in A}) for h in H}
def ev(h, span=None):
 f=factor if span is None else factor.loc[span[0]:span[1]]; y=fwd[h].reindex(f.index); z=[]; nn=[]; dropped=0
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   r=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(r): z.append(r);nn.append(len(q))
   else:dropped+=1
 if not z:return {'dates':0}
 z=np.array(z); return {'dates':len(z),'ic':float(z.mean()),'icir':float(z.mean()/z.std(ddof=1)),'hit':float((z>0).mean()),'mean_names':float(np.mean(nn)),'min_names':int(np.min(nn)),'dropped_constant':dropped}
print('FACTOR vix_regime_conditioned_risk_adjusted_trend_20 cutoff',cutoff.date())
print('CELLS',int(factor.notna().sum().sum()),'/',factor.size,'coverage',float(factor.notna().stack().mean()))
for h in H: print('H',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-02'))]: print('REGIME_5',n,ev(5,s))
print('TURNOVER',float(factor.rank(axis=1,pct=True).diff().abs().stack().mean()))
def quiet(a): return (C[a].pct_change(20).abs()/R[a].abs().rolling(20,min_periods=15).sum())*(1-R[a].rolling(20,min_periods=15).std().rolling(60,min_periods=40).apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1],raw=False))
lib={'relative_volume_participation':pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}),'risk_adjusted_trend':base,'volnorm_reversal_5':pd.DataFrame({a:-(C[a]/C[a].shift(5)-1)/R[a].rolling(5,min_periods=4).std() for a in A}),'volscaled_reversal_1':pd.DataFrame({a:-R[a]/R[a].rolling(20,min_periods=15).std() for a in A}),'quiet_trend_path_efficiency':pd.DataFrame({a:quiet(a) for a in A})}
mx=0; who=None
for n,x in lib.items():
 q=pd.concat([factor.stack(),x.stack()],axis=1).dropna(); rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic) if len(q)>2 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',rho)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'WITH',who)
