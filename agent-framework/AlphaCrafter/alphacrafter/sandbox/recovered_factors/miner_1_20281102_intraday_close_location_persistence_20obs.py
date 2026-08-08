"""Validate 20-observation intraday close-location persistence (one factor idea)."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2028-11-01')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 return d.loc[d.index<=CUT]
D={s:load(s) for s in AS}; dates=sorted(set().union(*[x.index for x in D.values()]))
# union panels preserve native missingness
C=pd.DataFrame({s:D[s].close for s in AS}).reindex(dates); R=C.pct_change()
H=pd.DataFrame({s:D[s].high for s in AS}).reindex(dates); L=pd.DataFrame({s:D[s].low for s in AS}).reindex(dates); O=pd.DataFrame({s:D[s].open for s in AS}).reindex(dates)
V=pd.DataFrame({s:D[s].volume for s in AS}).reindex(dates)
def rollcorr(a,b,n,mask=None):
 out=pd.DataFrame(index=a.index,columns=AS,dtype=float)
 for s in AS:
  aa=a[s]; bb=b if isinstance(b,pd.Series) else b[s]
  if mask is not None: aa=aa.where(mask); bb=bb.where(mask)
  out[s]=aa.rolling(n,min_periods=max(8,n//3)).corr(bb)
 return out
def beta(a,b,n,mask=None):
 out=pd.DataFrame(index=a.index,columns=AS,dtype=float)
 for s in AS:
  aa=a[s]; bb=b if isinstance(b,pd.Series) else b[s]
  if mask is not None: aa=aa.where(mask);bb=bb.where(mask)
  out[s]=aa.rolling(n,min_periods=max(10,n//3)).cov(bb)/bb.rolling(n,min_periods=max(10,n//3)).var()
 return out
med=R.median(axis=1); disp=R.std(axis=1); vol20=R.rolling(20,min_periods=12).std()
# candidate: bounded position of close in its completed daily range, averaged across 20 observations
rng=(H-L).replace(0,np.nan); CLV=(2*C-H-L)/rng
cand=CLV.rolling(20,min_periods=12).mean()
# library reconstructions, all non-backup admitted jsons
lib={}
lib['volnorm_reversal_5obs']=-R.rolling(5,min_periods=3).sum()/R.rolling(20,min_periods=12).std()
lib['correlation_asymmetry_60obs']=rollcorr(R,med,60,med<0)-rollcorr(R,med,60,med>=0)
lib['return_sign_balance_20obs']=(R>0).rolling(20,min_periods=12).mean()-(R<0).rolling(20,min_periods=12).mean()
lib['dispersion_sensitivity_20obs']=rollcorr(R,disp,20)
lib['volatility_clustering_autocorr_20obs']=pd.DataFrame({s:R[s].abs().rolling(20,min_periods=15).apply(lambda x: pd.Series(x).autocorr(),raw=False) for s in AS})
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(dates).ffill(); dv=vix.pct_change()
lib['adaptive_vix_relief_beta_change_25_60obs']=beta(R,dv,25,dv<0)-beta(R,dv,60,dv<0)
gap=O/C.shift(1)-1; day=C/O-1
lib['overnight_daytime_reversal_concordance_20obs']=(-(gap*day)).rolling(20,min_periods=12).mean()
lib['vix_shock_relief_beta_asymmetry_60obs']=beta(R,dv,60,dv>0)-beta(R,dv,60,dv<0)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.reindex(dates).ffill().pct_change()
trend=med.rolling(20,min_periods=12).mean()
lib['dxy_median_trend_regime_beta_spread_60obs']=beta(R,dxy,60,trend>0)-beta(R,dxy,60,trend<=0)
state=vol20.gt(vol20.rolling(60,min_periods=30).median())
lib['dxy_relative_vol_regime_beta_spread_60obs']=pd.DataFrame({s:beta(R,dxy,60,state[s])[s]-beta(R,dxy,60,~state[s])[s] for s in AS})
lib['realized_volatility_20obs']=vol20
# remaining active factors
b=beta(R,med,60); resid=R-b.mul(med,axis=0)
lib['residual_downside_semivol_share_60obs']=(resid.clip(upper=0).pow(2).rolling(60,min_periods=30).mean().pow(.5)/resid.pow(2).rolling(60,min_periods=30).mean().pow(.5))
lib['vol_orthogonal_median_beta_60obs']=b.sub(b.mean(axis=1),axis=0) # conservative proxy; correlation screen also compares beta structure
lib['excess_downside_beta_ca_orthogonal_60obs']=beta(R,med,60,med<0)-b
lib['downside_volume_participation_asymmetry_60obs']=np.log(V.where(R<0).rolling(60,min_periods=20).mean()/V.where(R>=0).rolling(60,min_periods=20).mean())
lib['inverted_downside_volume_participation_acceleration_20_60obs']=-(np.log(V.where(R<0).rolling(20,min_periods=8).mean()/V.where(R>=0).rolling(20,min_periods=8).mean())-np.log(V.where(R<0).rolling(60,min_periods=20).mean()/V.where(R>=0).rolling(60,min_periods=20).mean()))
range_rel=(H-L)/C
lib['inverted_dispersion_regime_range_state_20_60obs']=-np.log(range_rel.rolling(20,min_periods=12).mean()/range_rel.rolling(60,min_periods=30).mean())
lib['relative_volume_participation_20d']=np.log(V/V.rolling(20,min_periods=12).mean())
lib['risk_adjusted_trend_20d']=R.rolling(20,min_periods=12).sum()/vol20
lib['risk_adjusted_trend_acceleration_20_60d']=lib['risk_adjusted_trend_20d']-R.rolling(60,min_periods=30).sum()/R.rolling(60,min_periods=30).std()
lib['return_persistence_autocorr_20obs']=pd.DataFrame({s:R[s].rolling(20,min_periods=15).apply(lambda x:pd.Series(x).autocorr(),raw=False) for s in AS})
lib['return_directional_efficiency_20obs']=R.rolling(20,min_periods=12).sum().abs()/R.abs().rolling(20,min_periods=12).sum()
lib['relative_liquidity_stress_20_60obs']=np.log((R.abs()/V).rolling(20,min_periods=12).mean()/(R.abs()/V).rolling(60,min_periods=30).mean())
def icstats(h):
 fwd=C.shift(-h)/C-1; arr=[]; ns=[]
 for t in cand.index:
  x=cand.loc[t];y=fwd.loc[t]; z=x.notna()&y.notna()
  if z.sum()>=8: arr.append(spearmanr(x[z],y[z]).statistic);ns.append(z.sum())
 a=np.array(arr); return len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean()),float(np.mean(ns))
print('CANDIDATE intraday_close_location_persistence_20obs; cutoff',CUT.date())
for h in [1,5,10,20]: print('H',h,'dates IC ICIR hit meanN',icstats(h))
# rank turnover
print('coverage',cand.notna().sum().sum(), '/',cand.size, 'turnover',cand.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
# correlations: max cross-sectional Spearman by date, then absolute average (same definition used in prior validation)
res=[]
for name,x in lib.items():
 vals=[]
 for t in cand.index:
  a=cand.loc[t];q=x.loc[t]; z=a.notna()&q.notna()
  if z.sum()>=8: vals.append(spearmanr(a[z],q[z]).statistic)
 if vals: res.append((name,len(vals),float(np.mean(np.abs(vals))),float(np.max(np.abs(vals)))))
print('LIBRARY CORRELATIONS name dates mean_abs max_single')
for z in sorted(res,key=lambda x:-x[2]): print(z)
print('MAX_MEAN_ABS',max(x[2] for x in res))
# regimes based market 20d direction
for label,mask in [('up',trend>0),('down',trend<=0)]:
 vals=[]; f=C.shift(-5)/C-1
 for t in cand.index[mask.reindex(cand.index).fillna(False)]:
  z=cand.loc[t].notna()&f.loc[t].notna()
  if z.sum()>=8: vals.append(spearmanr(cand.loc[t][z],f.loc[t][z]).statistic)
 print('REGIME',label,'n',len(vals),'IC',np.mean(vals) if vals else None,'ICIR',np.mean(vals)/np.std(vals,ddof=1) if len(vals)>1 else None)
