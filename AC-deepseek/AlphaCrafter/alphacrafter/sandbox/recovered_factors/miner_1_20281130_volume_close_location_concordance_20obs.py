"""Validate volume--close-location concordance over 20 completed observations."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2028-11-29')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
D={s:load(s) for s in AS}; dates=sorted(set().union(*[x.index for x in D.values()]))
C=pd.DataFrame({s:D[s].close for s in AS}).reindex(dates);R=C.pct_change();H=pd.DataFrame({s:D[s].high for s in AS}).reindex(dates);L=pd.DataFrame({s:D[s].low for s in AS}).reindex(dates);O=pd.DataFrame({s:D[s].open for s in AS}).reindex(dates);V=pd.DataFrame({s:D[s].volume for s in AS}).reindex(dates)
def beta(a,b,n,mask=None):
 o=pd.DataFrame(index=a.index,columns=AS,dtype=float)
 for s in AS:
  aa=a[s];bb=b if isinstance(b,pd.Series) else b[s]
  if mask is not None:aa=aa.where(mask);bb=bb.where(mask)
  o[s]=aa.rolling(n,min_periods=max(10,n//3)).cov(bb)/bb.rolling(n,min_periods=max(10,n//3)).var()
 return o
def rcorr(a,b,n):return pd.DataFrame({s:a[s].rolling(n,min_periods=12).corr(b[s]) for s in AS})
med=R.median(axis=1);disp=R.std(axis=1);vol20=R.rolling(20,min_periods=12).std()
# Candidate: rolling concordance of abnormal volume with closing location inside each day's range.
# High values indicate participation-confirmed closes near the intraday high; low values indicate high-volume weak closes.
clv=((C-L)/(H-L)).where(H>L).clip(0,1)
vr=np.log(V/V.rolling(60,min_periods=30).median()).replace([np.inf,-np.inf],np.nan)
cand=rcorr(clv,vr,20)
lib={}
lib['volnorm_reversal_5obs']=-R.rolling(5,min_periods=3).sum()/vol20
lib['correlation_asymmetry_60obs']=pd.DataFrame({s:R[s].where(med<0).rolling(60,min_periods=20).corr(med.where(med<0))-R[s].where(med>=0).rolling(60,min_periods=20).corr(med.where(med>=0)) for s in AS})
lib['return_sign_balance_20obs']=(R>0).rolling(20,min_periods=12).mean()-(R<0).rolling(20,min_periods=12).mean()
lib['dispersion_sensitivity_20obs']=pd.DataFrame({s:R[s].rolling(20,min_periods=12).corr(disp) for s in AS})
lib['volatility_clustering_autocorr_20obs']=pd.DataFrame({s:R[s].abs().rolling(20,min_periods=15).apply(lambda x:pd.Series(x).autocorr(),raw=False) for s in AS})
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(dates).ffill();dv=vix.pct_change()
lib['adaptive_vix_relief_beta_change_25_60obs']=beta(R,dv,25,dv<0)-beta(R,dv,60,dv<0)
gap=O/C.shift(1)-1;day=C/O-1;lib['overnight_daytime_reversal_concordance_20obs']=(-(gap*day)).rolling(20,min_periods=12).mean();lib['vix_shock_relief_beta_asymmetry_60obs']=beta(R,dv,60,dv>0)-beta(R,dv,60,dv<0)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.reindex(dates).ffill().pct_change();trend=med.rolling(20,min_periods=12).mean()
lib['dxy_median_trend_regime_beta_spread_60obs']=beta(R,dxy,60,trend>0)-beta(R,dxy,60,trend<=0);state=vol20.gt(vol20.rolling(60,min_periods=30).median());lib['dxy_relative_vol_regime_beta_spread_60obs']=pd.DataFrame({s:beta(R,dxy,60,state[s])[s]-beta(R,dxy,60,~state[s])[s] for s in AS})
lib['realized_volatility_20obs']=vol20;b=beta(R,med,60);resid=R-b.mul(med,axis=0);lib['residual_downside_semivol_share_60obs']=resid.clip(upper=0).pow(2).rolling(60,min_periods=30).mean().pow(.5)/resid.pow(2).rolling(60,min_periods=30).mean().pow(.5);lib['vol_orthogonal_median_beta_60obs']=b.sub(b.mean(axis=1),axis=0);lib['excess_downside_beta_ca_orthogonal_60obs']=beta(R,med,60,med<0)-b
lib['downside_volume_participation_asymmetry_60obs']=np.log(V.where(R<0).rolling(60,min_periods=20).mean()/V.where(R>=0).rolling(60,min_periods=20).mean());lib['inverted_downside_volume_participation_acceleration_20_60obs']=-(np.log(V.where(R<0).rolling(20,min_periods=8).mean()/V.where(R>=0).rolling(20,min_periods=8).mean())-np.log(V.where(R<0).rolling(60,min_periods=20).mean()/V.where(R>=0).rolling(60,min_periods=20).mean()));rr=(H-L).div(C)
lib['inverted_dispersion_regime_range_state_20_60obs']=-np.log(rr.rolling(20,min_periods=12).mean()/rr.rolling(60,min_periods=30).mean());lib['relative_volume_participation_20d']=np.log(V/V.rolling(20,min_periods=12).mean());lib['risk_adjusted_trend_20d']=R.rolling(20,min_periods=12).sum()/vol20;lib['risk_adjusted_trend_acceleration_20_60d']=lib['risk_adjusted_trend_20d']-R.rolling(60,min_periods=30).sum()/R.rolling(60,min_periods=30).std();lib['return_persistence_autocorr_20obs']=pd.DataFrame({s:R[s].rolling(20,min_periods=15).apply(lambda x:pd.Series(x).autocorr(),raw=False) for s in AS});lib['return_directional_efficiency_20obs']=R.rolling(20,min_periods=12).sum().abs()/R.abs().rolling(20,min_periods=12).sum();lib['relative_liquidity_stress_20_60obs']=np.log((R.abs()/V).rolling(20,min_periods=12).mean()/(R.abs()/V).rolling(60,min_periods=30).mean())
def stat(h,mask=None):
 f=C.shift(-h)/C-1;z=[];ns=[]
 ix=cand.index if mask is None else cand.index[mask.reindex(cand.index).fillna(False)]
 for t in ix:
  ok=cand.loc[t].notna()&f.loc[t].notna()
  if ok.sum()>=8:z.append(spearmanr(cand.loc[t][ok],f.loc[t][ok]).statistic);ns.append(ok.sum())
 z=np.array(z);return len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),np.mean(ns)
print('CANDIDATE volume_close_location_concordance_20obs cutoff',CUT.date())
for h in [1,5,10,20]:print('H',h,'dates IC ICIR hit meanN',stat(h))
print('coverage',cand.notna().sum().sum(),'/',cand.size,'turnover',cand.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=[]
for name,q in lib.items():
 z=[]
 for t in cand.index:
  ok=cand.loc[t].notna()&q.loc[t].notna()
  if ok.sum()>=8:z.append(spearmanr(cand.loc[t][ok],q.loc[t][ok]).statistic)
 if z:out.append((name,len(z),float(np.mean(np.abs(z))),float(np.max(np.abs(z)))))
print('LIBRARY_CORRELATIONS name dates mean_abs max_single')
for x in sorted(out,key=lambda x:-x[3]):print(x)
print('MAX_ABS_LIBRARY_CORRELATION',max(x[3] for x in out))
for name,m in [('up',trend>0),('down',trend<=0)]:print('REGIME',name,'H5',stat(5,m))
