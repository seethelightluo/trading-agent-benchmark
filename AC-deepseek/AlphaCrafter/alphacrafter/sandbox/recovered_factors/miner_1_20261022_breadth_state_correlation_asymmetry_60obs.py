"""miner_1: validate breadth-state correlation asymmetry, a cross-asset stress factor."""
import numpy as np, pandas as pd, glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; V={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 P[a]=d.close.astype(float); V[a]=d.volume.astype(float)
P=pd.DataFrame(P); V=pd.DataFrame(V); R=P.pct_change()
# On each day breadth is fraction of tradable assets with positive return. Signal = correlation with tape on weak-breadth days minus strong-breadth days, last 60 observations.
tape=R.median(axis=1); breadth=(R>0).mean(axis=1)
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 pair=pd.concat([R[a].rename('r'),tape.rename('t'),breadth.rename('b')],axis=1)
 for i,dt in enumerate(pair.index):
  z=pair.iloc[max(0,i-59):i+1].dropna()
  lo=z[z.b<=.4]; hi=z[z.b>=.6]
  if len(lo)>=12 and len(hi)>=12: F.loc[dt,a]=lo.r.corr(lo.t)-hi.r.corr(hi.t)
L={}
L['miner_3_risk_adjusted_trend_20d']=(P/P.shift(20)-1)/R.rolling(20,min_periods=15).std()
L['miner_3_relative_volume_participation_20d']=np.log(V/V.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
L['miner_1_volnorm_reversal_5obs']=-(P/P.shift(5)-1)/R.rolling(5,min_periods=4).std()
L['miner_2_realized_volatility_20obs']=R.rolling(20,min_periods=15).std()
L['cross_asset_beta_compression_20obs']=R.rolling(20,min_periods=15).corr(tape)
L['risk_adjusted_trend_acceleration_20_60d']=L['miner_3_risk_adjusted_trend_20d']-(P/P.shift(60)-1)/R.rolling(60,min_periods=45).std()
# exact recently-admitted conditional-correlation factor
C=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 pair=pd.concat([R[a].rename('r'),tape.rename('t')],axis=1)
 for i,dt in enumerate(pair.index):
  z=pair.iloc[max(0,i-59):i+1].dropna(); lo=z[z.t<0]; hi=z[z.t>=0]
  if len(lo)>=12 and len(hi)>=12: C.loc[dt,a]=lo.r.corr(lo.t)-hi.r.corr(hi.t)
L['miner_1_correlation_asymmetry_60obs']=C
print('FACTOR breadth_state_correlation_asymmetry_60obs = corr(asset,tape | breadth<=0.4)-corr(asset,tape | breadth>=0.6), trailing 60d')
print('visible_through',P.index.max().date(),'history',P.index.min().date(),P.index.max().date(),'instruments',len(A),'cells',int(F.notna().sum().sum()),'coverage',f'{F.notna().mean().mean():.4f}')
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; out=[]; cov=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: out.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(out)); sd=x.std(ddof=1)
 print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} xs_coverage={np.mean(cov):.4f}')
 if h==10:
  for n,m in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
   q=x[m];print(f' REGIME {n} n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=F.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('rank_turnover',f'{np.mean(ts):.6f}')
mx=0
for n,X in L.items():
 z=pd.concat([F.stack().rename('f'),X.stack().rename('l')],axis=1).dropna();rho=z.f.corr(z.l,method='spearman');mx=max(mx,abs(rho));print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; json_count={len([x for x in glob.glob("factors/*.json") if not x.endswith(".bak")])}')
