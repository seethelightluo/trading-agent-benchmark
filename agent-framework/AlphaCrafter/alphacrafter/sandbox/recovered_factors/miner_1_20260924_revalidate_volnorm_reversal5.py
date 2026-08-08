"""Quarterly revalidation using each asset's native daily sequence."""
import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={};FW={};L={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();p=d.close.astype(float);r=p.pct_change();v=d.volume.astype(float)
 F[a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();FW[a]={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
 L.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std();L.setdefault('miner_3_relative_volume_participation_20d',{})[a]=np.log(v/v.rolling(20,min_periods=15).mean());L.setdefault('miner_2_realized_volatility_20obs',{})[a]=r.rolling(20,min_periods=15).std();L.setdefault('risk_adjusted_trend_acceleration_20_60d',{})[a]=L['miner_3_risk_adjusted_trend_20d'][a]-(p/p.shift(60)-1)/r.rolling(60,min_periods=45).std()
prices=pd.DataFrame({a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close for a in A});rr=prices.pct_change();med=rr.median(axis=1)
for a in A:L.setdefault('cross_asset_beta_compression_20obs',{})[a]=rr[a].rolling(20,min_periods=15).corr(med)
f=pd.DataFrame(F).sort_index();print('FACTOR revalidation: five-observation volatility-normalized reversal');print('history',f.index.min().date(),f.index.max().date(),'instruments',len(A))
R={}
for h in [1,5,10,20]:
 fw=pd.DataFrame({a:FW[a][h] for a in A});q=[];cv=[]
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append((dt,z.x.corr(z.y,method='spearman')));cv.append(len(z)/15)
 x=pd.Series(dict(q));R[h]=x;s=x.std(ddof=1);print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/s:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cv):.4f}')
 if h==1:
  for n,m in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
   y=x[m];print(f' {n}: dates={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
r=f.rank(axis=1,pct=True);t=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(t):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=0
for n,g in L.items():
 z=pd.concat([f.stack().rename('x'),pd.DataFrame(g).stack().rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.x.corr(z.y,method='spearman');mx=max(mx,abs(rho));print(f'LIB {n}: rho={rho:.6f}, cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; library_records={len(glob.glob("factors/*.json"))}')
