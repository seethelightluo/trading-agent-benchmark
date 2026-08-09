"""One candidate: idiosyncratic-return / volume-surprise elasticity, with full library independence audit."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2028-12-13')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
D={s:load(s) for s in AS}; dates=sorted(set().union(*[x.index for x in D.values()])); C=pd.DataFrame({s:D[s].close for s in AS}).reindex(dates); O=pd.DataFrame({s:D[s].open for s in AS}).reindex(dates); H=pd.DataFrame({s:D[s].high for s in AS}).reindex(dates); L=pd.DataFrame({s:D[s].low for s in AS}).reindex(dates); V=pd.DataFrame({s:D[s].volume for s in AS}).reindex(dates); R=C.pct_change(); med=R.median(axis=1); disp=R.std(axis=1); vol20=R.rolling(20,min_periods=12).std()
def beta(a,b,n,mask=None):
 out=pd.DataFrame(index=a.index,columns=AS,dtype=float)
 for s in AS:
  x=a[s]; y=b if isinstance(b,pd.Series) else b[s]
  if mask is not None:x=x.where(mask);y=y.where(mask)
  out[s]=x.rolling(n,min_periods=max(10,n//3)).cov(y)/y.rolling(n,min_periods=max(10,n//3)).var()
 return out
# Higher reading: volume surprise is consistently associated with positive asset-specific, rather than market, returns.
b=beta(R,med,60); resid=R-b.mul(med,axis=0); vs=np.log(V/V.rolling(60,min_periods=30).median()).replace([np.inf,-np.inf],np.nan)
cand=pd.DataFrame({s:resid[s].rolling(60,min_periods=30).corr(vs[s]) for s in AS})
def rcorr(a,b,n): return pd.DataFrame({s:a[s].rolling(n,min_periods=12).corr(b[s]) for s in AS})
MM=pd.DataFrame({s:med for s in AS}); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(dates).ffill();dv=vix.pct_change(); dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.reindex(dates).ffill().pct_change(); trend=med.rolling(20,min_periods=12).mean()
gap=O/C.shift(1)-1; day=C/O-1; rr=(H-L)/C
lib={
'volnorm_reversal_5obs':-R.rolling(5,min_periods=3).sum()/vol20,
'correlation_asymmetry_60obs':pd.DataFrame({s:R[s].where(med<0).rolling(60,min_periods=20).corr(med.where(med<0))-R[s].where(med>=0).rolling(60,min_periods=20).corr(med.where(med>=0)) for s in AS}),
'return_sign_balance_20obs':(R>0).rolling(20,min_periods=12).mean()-(R<0).rolling(20,min_periods=12).mean(),
'dispersion_sensitivity_20obs':pd.DataFrame({s:R[s].rolling(20,min_periods=12).corr(disp) for s in AS}),
'volatility_clustering_autocorr_20obs':pd.DataFrame({s:R[s].abs().rolling(20,min_periods=15).apply(lambda x:pd.Series(x).autocorr(),raw=False) for s in AS}),
'adaptive_vix_relief_beta_change_25_60obs':beta(R,dv,25,dv<0)-beta(R,dv,60,dv<0),
'overnight_daytime_reversal_concordance_20obs':(-(gap*day)).rolling(20,min_periods=12).mean(),
'vix_shock_relief_beta_asymmetry_60obs':beta(R,dv,60,dv>0)-beta(R,dv,60,dv<0),
'dxy_median_trend_regime_beta_spread_60obs':beta(R,dxy,60,trend>0)-beta(R,dxy,60,trend<=0),
'dxy_relative_vol_regime_beta_spread_60obs':pd.DataFrame({s:beta(R,dxy,60,vol20[s]>vol20[s].rolling(60,min_periods=30).median())[s]-beta(R,dxy,60,vol20[s]<=vol20[s].rolling(60,min_periods=30).median())[s] for s in AS}),
'realized_volatility_20obs':vol20,
'residual_downside_semivol_share_60obs':resid.clip(upper=0).pow(2).rolling(60,min_periods=30).mean().pow(.5)/resid.pow(2).rolling(60,min_periods=30).mean().pow(.5),
'vol_orthogonal_median_beta_60obs':b.sub(b.mean(axis=1),axis=0),'excess_downside_beta_ca_orthogonal_60obs':beta(R,med,60,med<0)-b,
'downside_volume_participation_asymmetry_60obs':np.log(V.where(R<0).rolling(60,min_periods=20).mean()/V.where(R>=0).rolling(60,min_periods=20).mean()),
'inverted_downside_volume_participation_acceleration_20_60obs':-(np.log(V.where(R<0).rolling(20,min_periods=8).mean()/V.where(R>=0).rolling(20,min_periods=8).mean())-np.log(V.where(R<0).rolling(60,min_periods=20).mean()/V.where(R>=0).rolling(60,min_periods=20).mean())),
'inverted_dispersion_regime_range_state_20_60obs':-np.log(rr.rolling(20,min_periods=12).mean()/rr.rolling(60,min_periods=30).mean()),'relative_volume_participation_20d':np.log(V/V.rolling(20,min_periods=12).mean()),'risk_adjusted_trend_20d':R.rolling(20,min_periods=12).sum()/vol20,'risk_adjusted_trend_acceleration_20_60d':R.rolling(20,min_periods=12).sum()/vol20-R.rolling(60,min_periods=30).sum()/R.rolling(60,min_periods=30).std(),'return_persistence_autocorr_20obs':pd.DataFrame({s:R[s].rolling(20,min_periods=15).apply(lambda x:pd.Series(x).autocorr(),raw=False) for s in AS}),'return_directional_efficiency_20obs':R.rolling(20,min_periods=12).sum().abs()/R.abs().rolling(20,min_periods=12).sum(),'relative_liquidity_stress_20_60obs':np.log((R.abs()/V).rolling(20,min_periods=12).mean()/(R.abs()/V).rolling(60,min_periods=30).mean())}
def stat(h,subset=None):
 fw=C.shift(-h)/C-1; vals=[]; ns=[]; use=cand.index if subset is None else cand.index[subset.reindex(cand.index).fillna(False)]
 for t in use:
  ok=cand.loc[t].notna()&fw.loc[t].notna()
  if ok.sum()>=8: vals.append(spearmanr(cand.loc[t,ok],fw.loc[t,ok]).statistic);ns.append(ok.sum())
 x=np.array(vals); return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)
print('CANDIDATE idiosyncratic_return_volume_elasticity_60obs cutoff',CUT.date(),'actual',C.index.max().date(),'assets',len(AS))
for h in [1,5,10,20]:print('H',h,'dates IC ICIR hit meanN',stat(h))
print('coverage',int(cand.notna().sum().sum()),'/',cand.size,'rank_turnover',cand.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n,mask in [('2020_21',(C.index<'2022-01-01')),('2022_23',(C.index>='2022-01-01')&(C.index<'2024-01-01')),('2024_25',(C.index>='2024-01-01')&(C.index<'2026-01-01')),('2026_current',C.index>='2026-01-01')]:print('REGIME',n,'H10',stat(10,mask))
allrho=[]
for n,q in lib.items():
 z=[]
 for t in cand.index:
  ok=cand.loc[t].notna()&q.loc[t].notna()
  if ok.sum()>=8:z.append(spearmanr(cand.loc[t,ok],q.loc[t,ok]).statistic)
 if not z: print('LIB',n,'MISSING'); raise RuntimeError('missing required correlation evidence')
 m=float(np.max(np.abs(z)));allrho.append((m,n,len(z),float(np.mean(z)))); print('LIB',n,'dates',len(z),'mean_rho',round(np.mean(z),6),'max_abs',round(m,6))
print('MAX_ABS_LIBRARY_CORRELATION',max(allrho))
