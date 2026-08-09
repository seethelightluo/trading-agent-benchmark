"""miner_1: validate serial persistence in relative-volume deviations."""
import glob, numpy as np, pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p): return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
P={};V={};R={};FW={};LIB={k:{} for k in ['risk_adjusted_trend_20d','relative_volume_participation_20d','volnorm_reversal_5obs','realized_volatility_20obs','cross_asset_beta_compression_20obs','risk_adjusted_trend_acceleration_20_60d','correlation_asymmetry_60obs','return_skewness_20obs']}
for a in ASSETS:
 d=load(f'../persistent/stock_data/{a}.csv');p=d.close.astype(float);v=d.volume.astype(float);r=p.pct_change();P[a]=p;V[a]=v;R[a]=r
 rel=np.log(v/v.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
 # high score means relative participation has persistent, rather than isolated, deviations
 F=rel.rolling(20,min_periods=15).corr(rel.shift(1))
 LIB['relative_volume_participation_20d'][a]=rel
 fast=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std();slow=(p/p.shift(60)-1)/r.rolling(60,min_periods=45).std()
 LIB['risk_adjusted_trend_20d'][a]=fast;LIB['risk_adjusted_trend_acceleration_20_60d'][a]=fast-slow
 LIB['volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();LIB['realized_volatility_20obs'][a]=r.rolling(20,min_periods=15).std();LIB['return_skewness_20obs'][a]=r.rolling(20,min_periods=15).skew()
 FW[a]={h:p.shift(-h)/p-1 for h in (1,5,10,20)}
 if a==ASSETS[0]: f=pd.DataFrame(index=p.index)
 f[a]=F
rr=pd.DataFrame(R);med=rr.median(axis=1)
for a in ASSETS:
 LIB['cross_asset_beta_compression_20obs'][a]=rr[a].rolling(20,min_periods=15).corr(med)
 vals=[]
 for i in range(len(rr)):
  x=rr[a].iloc[max(0,i-59):i+1];y=med.iloc[max(0,i-59):i+1];dn=x[y<0];up=x[y>=0]
  vals.append(dn.corr(y[y<0])-up.corr(y[y>=0]) if len(dn)>=12 and len(up)>=12 else np.nan)
 LIB['correlation_asymmetry_60obs'][a]=pd.Series(vals,index=rr.index)
print('FACTOR: relative_volume_serial_persistence_20obs = rolling_corr_20(log(V/mean20(V)), lag1(log(V/mean20(V))))')
print('history',f.index.min().date(),f.index.max().date(),'assets',len(ASSETS))
def ic(h):
 fw=pd.DataFrame({a:FW[a][h] for a in ASSETS});out=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:out.append((dt,z.f.corr(z.r,method='spearman')));cov.append(len(z)/15)
 return pd.Series(dict(out)),np.mean(cov)
for h in (1,5,10,20):
 x,c=ic(h);sd=x.std(ddof=1);print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for n,m in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-current',x.index>='2025-01-01')]:
  q=x[m];print(f' {n} n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(turn):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=0
for n,d in LIB.items():
 z=pd.concat([f.stack().rename('new'),pd.DataFrame(d).stack().rename('old')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print(f'LIB {n}: rho={rho:.6f}, cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; library_json_count={len(glob.glob("factors/*.json"))}')
