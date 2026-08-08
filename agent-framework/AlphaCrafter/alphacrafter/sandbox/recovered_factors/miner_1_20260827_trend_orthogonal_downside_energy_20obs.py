"""miner_1: validate trend-orthogonal downside energy share (20 observations)."""
import glob
import numpy as np
import pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}; trend={}; fw={}; lib={k:{} for k in ['miner_3_risk_adjusted_trend_20d','miner_3_relative_volume_participation_20d','miner_1_volnorm_reversal_5obs','miner_2_realized_volatility_20obs']}
for a in ASSETS:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index(); p=d.close.astype(float); r=p.pct_change()
 raw[a]=(r.clip(upper=0).pow(2).rolling(20,min_periods=15).sum()/r.pow(2).rolling(20,min_periods=15).sum()).replace([np.inf,-np.inf],np.nan)
 trend[a]=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
 fw[a]={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
 lib['miner_3_risk_adjusted_trend_20d'][a]=trend[a]
 lib['miner_3_relative_volume_participation_20d'][a]=np.log(d.volume.astype(float)/d.volume.astype(float).rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
 lib['miner_1_volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
 lib['miner_2_realized_volatility_20obs'][a]=r.rolling(20,min_periods=15).std()
raw=pd.DataFrame(raw).sort_index(); tr=pd.DataFrame(trend).reindex(raw.index)
# Each date: residual from cross-sectional OLS of downside energy share on risk-adjusted 20d trend, including intercept.
f=pd.DataFrame(np.nan,index=raw.index,columns=ASSETS)
for dt in f.index:
 z=pd.concat([raw.loc[dt].rename('y'),tr.loc[dt].rename('x')],axis=1).dropna()
 if len(z)>=8:
  b=np.linalg.lstsq(np.c_[np.ones(len(z)),z.x.values],z.y.values,rcond=None)[0]
  f.loc[dt,z.index]=z.y-(b[0]+b[1]*z.x)
print('FACTOR trend_orthogonal_downside_energy_20obs = cross-sectional OLS residual of downside squared-return energy share (20 obs, min 15) after 20d risk-adjusted trend')
print('history',f.index.min().date(),f.index.max().date(),'assets',len(ASSETS))
def ic(h):
 y=pd.DataFrame({a:fw[a][h] for a in ASSETS}); vals=[]; cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 return pd.Series(dict(vals)),np.mean(cov)
for h in [1,5,10,20]:
 x,c=ic(h); sd=x.std(ddof=1)
 print(f'H={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} absIC={abs(x.mean()):.6f} absICIR={abs(x.mean()/sd):.6f} hit={(x>0).mean():.4f} se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for n,m in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
  q=x[m]; print(f'  {n}: n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True); to=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=0
for n,v in lib.items():
 z=pd.concat([f.stack().rename('new'),pd.DataFrame(v).stack().rename('old')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print(f'LIB {n}: rho={rho:.6f}, cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; library_json_count={len(glob.glob("factors/*.json"))}')
