import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=[1,5,10,20]; cutoff=pd.Timestamp('2027-10-06')
C={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date'); C[a]=d.close.replace(0,np.nan)
px=pd.DataFrame(C); r=px.pct_change(); disp=r.std(axis=1,ddof=1)
# One idea: dispersion-exposure resilience.  Low exposure to broad cross-asset dispersion
# (correlation of absolute own moves with prior contemporaneous dispersion) should be defensively persistent.
f=-pd.DataFrame({a:r[a].abs().rolling(40,min_periods=25).corr(disp) for a in A}).shift(1)
fw={h:px.shift(-h).div(px)-1 for h in H}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=fw[h].reindex(x.index);z=[];n=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);n.append(len(q))
 z=np.array(z);return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),4),'mean_n':round(np.mean(n),2),'min_n':int(np.min(n))}
print('FACTOR inverse_absolute_return_dispersion_exposure_40 cutoff',cutoff.date(),'assets',len(A));print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().stack().mean(),5))
for h in H: print('H',h,ev(h))
for nm,sp in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027YTD',('2027-01-01','2027-10-06'))]:print('REGIME10',nm,ev(10,sp))
print('TURNOVER',round(f.rank(axis=1,pct=True).diff().abs().stack().mean(),6))
# Exact/reasonably literal library signal reconstitution; values compared on pooled date-asset observations.
m=r.median(axis=1); other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A});S={}
vol20=r.rolling(20,min_periods=15).std(); trend=(px/px.shift(20)-1)/vol20
S['gradual_volatility_contraction_gated_trend_20']=trend*np.tanh(np.clip(-np.log(vol20/ r.rolling(40,min_periods=20).std()),-2,2));S['ravmom_20obs']=trend;S['risk_adjusted_trend_20d']=trend
S['relative_volume_participation_20d']=None
S['quiet_trend_path_efficiency_20_60']=(px/px.shift(20)-1).abs()/r.abs().rolling(20,min_periods=15).sum()*(1-vol20.rolling(60,min_periods=40).rank(pct=True))
S['inverse_idiosyncratic_volatility_20']=-r.sub(m,axis=0).rolling(20,min_periods=15).std();S['downside_event_excess_magnitude_median_40']=pd.DataFrame({a:(r[a]-m).where(m<m.rolling(60,min_periods=40).quantile(.35).shift(1)).rolling(40,min_periods=1).median() for a in A}).shift(1)
S['low_commonality_other_median_correlation_40']=-pd.DataFrame({a:r[a].rolling(40,min_periods=25).corr(other[a]) for a in A});co=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(other[a]) for a in A});S['commonality_expansion_transition_40']=co.rolling(20,min_periods=15).mean()-co.shift(20).rolling(20,min_periods=15).mean()
S['downside_cross_asset_beta_resilience_40']=pd.DataFrame({a:r[a].where(m<0).rolling(40,min_periods=12).cov(m.where(m<0))/m.where(m<0).rolling(40,min_periods=12).var() for a in A});S['inverse_lag1_return_autocorrelation_20']=-pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(r[a].shift(1)) for a in A}); ac=-pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(r[a].shift(1)) for a in A});S['volatility_transition_serial_resilience_20']=ac*np.clip(np.log(r.rolling(5,min_periods=4).std()/vol20),-2,2)
S['inverse_lower_tail_persistence_40_60']=-pd.DataFrame({a:r[a].lt(r[a].rolling(60,min_periods=40).quantile(.2).shift(1)).rolling(40,min_periods=25).mean() for a in A});S['volnorm_reversal_5obs']=-(px/px.shift(5)-1)/r.rolling(5,min_periods=4).std();S['return_skewness_60']=r.rolling(60,min_periods=40).skew();S['volscaled_reversal_1obs']=-r/vol20
# Beta asymmetry exact approximate masking
S['downside_upside_cross_asset_beta_asymmetry_60']=pd.DataFrame({a:r[a].where(m<0).rolling(60,min_periods=15).cov(m.where(m<0))/m.where(m<0).rolling(60,min_periods=15).var()-r[a].where(m>=0).rolling(60,min_periods=15).cov(m.where(m>=0))/m.where(m>=0).rolling(60,min_periods=15).var() for a in A})
rows=[]
for name,g in S.items():
 if g is None:continue
 q=pd.concat([f.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic;rows.append((name,len(q),rho))
print('LIBRARY_CORRELATIONS (volume/VIX macro signals unavailable in this candidate script: admission requires exact follow-up if otherwise passing)')
for z in sorted(rows,key=lambda x:abs(x[2]),reverse=True):print(z[0],z[1],round(z[2],6))
print('MAX_RECONSTRUCTED',round(max(abs(z[2]) for z in rows),6))
