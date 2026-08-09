"""miner_1: correlation-asymmetry defensive-demand factor validation, 2026-10-08.
High signal: asset co-moves much more with the cross-asset tape on tape-down days than tape-up days."""
import glob, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-10-07') # last completed daily session visible at decision time
raw={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END] for a in A}
P=pd.DataFrame({a:x.close.astype(float) for a,x in raw.items()}).sort_index(); R=P.pct_change(); tape=R.median(axis=1)
def cc(x,y,isdown):
 out=[]
 for i in range(len(x)):
  z=pd.concat([x.iloc[max(0,i-59):i+1],y.iloc[max(0,i-59):i+1]],axis=1).dropna()
  z=z[z.iloc[:,1]<0] if isdown else z[z.iloc[:,1]>=0]
  out.append(z.iloc[:,0].corr(z.iloc[:,1]) if len(z)>=12 else np.nan)
 return pd.Series(out,index=x.index)
F=pd.DataFrame({a:cc(R[a],tape,True)-cc(R[a],tape,False) for a in A})
LIB={k:pd.DataFrame(index=P.index,columns=A,dtype=float) for k in ['miner_3_risk_adjusted_trend_20d','miner_3_relative_volume_participation_20d','miner_1_volnorm_reversal_5obs','miner_2_realized_volatility_20obs','cross_asset_beta_compression_20obs','risk_adjusted_trend_acceleration_20_60d']}
M=R.mean(axis=1)
for a in A:
 v=raw[a].volume.astype(float)
 LIB['miner_3_risk_adjusted_trend_20d'][a]=(P[a]/P[a].shift(20)-1)/R[a].rolling(20,min_periods=15).std()
 LIB['miner_3_relative_volume_participation_20d'][a]=np.log(v/v.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
 LIB['miner_1_volnorm_reversal_5obs'][a]=-(P[a]/P[a].shift(5)-1)/R[a].rolling(5,min_periods=4).std()
 LIB['miner_2_realized_volatility_20obs'][a]=R[a].rolling(20,min_periods=15).std()
 LIB['cross_asset_beta_compression_20obs'][a]=R[a].rolling(20,min_periods=15).cov(M)/M.rolling(20,min_periods=15).var()
 LIB['risk_adjusted_trend_acceleration_20_60d'][a]=((P[a]/P[a].shift(20)-1)-(P[a]/P[a].shift(60)-1))/R[a].rolling(20,min_periods=15).std()
print('FACTOR correlation_asymmetry_60obs; visible',P.index.min().date(),P.index.max().date(),'assets',len(A),'cells',F.notna().sum().sum())
def ic(h):
 y=P.shift(-h)/P-1; q=[]; cov=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 return pd.Series(dict(q)),np.mean(cov)
for h in [1,5,10,20]:
 x,c=ic(h); sd=x.std(ddof=1)
 print(f'H={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} coverage={c:.4f} instruments={15*c:.2f}')
 for n,m in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
  z=x[m];print(f' {n} n={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
r=F.rank(axis=1,pct=True); ts=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('turnover=%.6f coverage_all=%.4f'%(np.mean(ts),F.notna().mean().mean()))
mx=0
for n,L in LIB.items():
 z=pd.concat([F.stack().rename('f'),L.stack().rename('l')],axis=1).dropna(); rho=z.f.corr(z.l,method='spearman');mx=max(mx,abs(rho));print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; library_json_count={len(glob.glob("factors/*.json"))}')
