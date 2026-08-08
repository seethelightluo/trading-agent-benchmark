"""miner_3: validate one factor -- cross-asset SPX beta defensiveness."""
import pandas as pd, numpy as np, glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=[1,5,10,20]
P={}; R={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index(); P[a]=d.close.astype(float); R[a]=P[a].pct_change()
r=pd.DataFrame(R); p=pd.DataFrame(P)
# Negative 20-observation beta to the broad US equity benchmark: higher = less equity-sensitive / more defensive.
var=r['SPX'].rolling(20,min_periods=15).var().replace(0,np.nan)
f=-r.rolling(20,min_periods=15).cov(r['SPX']).div(var,axis=0)
fw={h:p.shift(-h).div(p)-1 for h in H}
# Reconstruct all currently admitted library signals faithfully from definitions.
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
rv=np.log(pd.DataFrame({a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().volume.astype(float).replace(0,np.nan) for a in A}) / pd.DataFrame({a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().volume.astype(float).replace(0,np.nan) for a in A}).rolling(20,min_periods=15).mean())
vol=r.rolling(20,min_periods=15).std()
acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol
orth=pd.DataFrame(index=p.index,columns=A,dtype=float)
for dt in p.index:
 z=pd.concat([acc.loc[dt].rename('a'),trend.loc[dt].rename('t')],axis=1).dropna()
 if len(z)>=8:
  b=np.polyfit(z.t,z.a,1);orth.loc[dt,z.index]=z.a-(b[0]*z.t+b[1])
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_3_relative_volume_participation_20d':rv,'miner_2_realized_volatility_20obs':vol,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':rev,'miner_3_orthogonal_trend_acceleration_20_60obs':orth}
print('FACTOR negative_spx_beta_20obs = -cov_20(asset return, SPX return)/var_20(SPX return); high scores are cross-asset equity defensiveness')
print('history',f.index.min().date(),f.index.max().date(),'assets',len(A))
def ic(h):
 o=[];cv=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),fw[h].loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:o.append((dt,z.x.corr(z.y,method='spearman')));cv.append(len(z)/15)
 return pd.Series(dict(o)),np.mean(cv)
for h in H:
 x,c=ic(h); print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} IC_se={x.std(ddof=1)/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for n,m in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  y=x[m]; print(f'  {n}: n={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);q=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:q.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'mean_rank_turnover={np.mean(q):.6f}; signal_cells={f.notna().sum().sum()}/{f.size} ({f.notna().mean().mean():.4f})')
mx=0
for n,l in lib.items():
 z=pd.concat([f.stack().rename('x'),l.stack().rename('y')],axis=1).dropna(); rho=z.x.corr(z.y,method='spearman');mx=max(mx,abs(rho));print(f'library_{n}_rho={rho:.6f}; cells={len(z)}')
print(f'library_files={len(glob.glob("factors/*.json"))}; max_abs_library_correlation={mx:.6f}')
