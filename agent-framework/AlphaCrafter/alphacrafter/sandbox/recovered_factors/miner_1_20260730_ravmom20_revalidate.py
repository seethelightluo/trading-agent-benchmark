"""miner_1 revalidation: 20-observation risk-adjusted momentum, using data visible through current cursor."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}; R={}; L={k:{} for k in ['miner_3_risk_adjusted_trend_20d','miner_3_relative_volume_participation_20d','miner_1_volnorm_reversal_5obs','miner_2_realized_volatility_20obs']}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index(); p=d.close.astype(float); r=p.pct_change()
 # Cross-asset trend strength: 20-day return per its own 20-day realized volatility.
 F[a]=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std(); R[a]=p
 L['miner_3_risk_adjusted_trend_20d'][a]=F[a]
 L['miner_3_relative_volume_participation_20d'][a]=np.log(d.volume.astype(float)/d.volume.astype(float).rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
 L['miner_1_volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
 L['miner_2_realized_volatility_20obs'][a]=r.rolling(20,min_periods=15).std()
f=pd.DataFrame(F); p=pd.DataFrame(R); print('FACTOR ravmom_20obs; range',f.index.min().date(),f.index.max().date(),'instruments',len(A))
primary=None
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; obs=[]; cov=[]
 for dt in f.index:
  z=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: obs.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(obs)); sd=x.std(ddof=1)
 print(f'H={h} n={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} se={sd/len(x)**.5:.6f} cov={np.mean(cov):.4f}')
 for n,m in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  q=x[m];print(f'  {n}: n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
 if h==10: primary=(x,np.mean(cov))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'TURNOVER={np.mean(ts):.6f} CELLCOVER={f.notna().mean().mean():.4f}')
for name,v in L.items():
 z=pd.concat([f.stack().rename('new'),pd.DataFrame(v).stack().rename('old')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); print(f'LIB {name}: rho={z.new.corr(z.old,method="spearman"):.6f} cells={len(z)}')
