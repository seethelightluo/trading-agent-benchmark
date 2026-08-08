"""miner_1: validate volatility-adjusted relative-volume consistency (single liquidity-path idea)."""
import glob, numpy as np, pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(path): return pd.read_csv(path,parse_dates=['date']).set_index('date').sort_index()
R={}; F={}; FW={}; LIB={k:{} for k in ['risk_adjusted_trend_20d','relative_volume_participation_20d','volnorm_reversal_5obs','realized_volatility_20obs','cross_asset_beta_compression_20obs','risk_adjusted_trend_acceleration_20_60d','correlation_asymmetry_60obs','return_skewness_20obs']}
for a in ASSETS:
 d=load(f'../persistent/stock_data/{a}.csv'); p=d.close.astype(float); v=d.volume.astype(float); r=p.pct_change(); R[a]=r
 rel=np.log(v/v.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
 rv=r.rolling(20,min_periods=15).std()
 # High score identifies predictable relative participation after normalizing its variation by native price volatility.
 F[a]=(-rel.rolling(20,min_periods=15).std()/rv).replace([np.inf,-np.inf],np.nan)
 FW[a]={h:p.shift(-h)/p-1 for h in (1,5,10,20)}
 fast=(p/p.shift(20)-1)/rv; slow=(p/p.shift(60)-1)/r.rolling(60,min_periods=45).std()
 LIB['risk_adjusted_trend_20d'][a]=fast; LIB['relative_volume_participation_20d'][a]=rel
 LIB['volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); LIB['realized_volatility_20obs'][a]=rv
 LIB['risk_adjusted_trend_acceleration_20_60d'][a]=fast-slow; LIB['return_skewness_20obs'][a]=r.rolling(20,min_periods=15).skew()
rr=pd.DataFrame(R); med=rr.median(axis=1)
for a in ASSETS:
 LIB['cross_asset_beta_compression_20obs'][a]=rr[a].rolling(20,min_periods=15).corr(med)
 vals=[]
 for i in range(len(rr)):
  x=rr[a].iloc[max(0,i-59):i+1]; y=med.iloc[max(0,i-59):i+1]; dn=x[y<0]; up=x[y>=0]
  vals.append(dn.corr(y[y<0])-up.corr(y[y>=0]) if len(dn)>=12 and len(up)>=12 else np.nan)
 LIB['correlation_asymmetry_60obs'][a]=pd.Series(vals,index=rr.index)
f=pd.DataFrame(F).sort_index(); print('FACTOR: volatility_adjusted_relative_volume_consistency_20obs = -std_20(log(volume/mean_20(volume)))/std_20(return)')
print('history',f.index.min().date(),f.index.max().date(),'assets',len(ASSETS))
def getic(h):
 fw=pd.DataFrame({a:FW[a][h] for a in ASSETS}); out=[]; cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: out.append((dt,z.f.corr(z.y,method='spearman'))); cov.append(len(z)/15)
 return pd.Series(dict(out)),np.mean(cov)
for h in (1,5,10,20):
 x,c=getic(h); sd=x.std(ddof=1); print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for nm,mask in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-current',x.index>='2025-01-01')]:
  q=x[mask]; print(f' {nm} n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(ts):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=0
for nm,d in LIB.items():
 z=pd.concat([f.stack().rename('new'),pd.DataFrame(d).stack().rename('old')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); rho=z.new.corr(z.old,method='spearman'); mx=max(mx,abs(rho)); print(f'LIB {nm}: rho={rho:.6f}, cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; library_json_count={len(glob.glob("factors/*.json"))}')
