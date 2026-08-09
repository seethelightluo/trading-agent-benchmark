"""miner_1: validate 20-observation drawdown persistence; no data later than 2026-12-16."""
import glob
import numpy as np
import pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-16'); P={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 P[a]=d.loc[:cut,'close'].astype(float)
p=pd.DataFrame(P); r=p.pct_change(); med=r.median(axis=1)
# Negative current peak-to-trough drawdown divided by observations since the 20-day peak.
# More negative values denote a deeper drawdown that has persisted rather than an abrupt dip.
def dd_persist(s):
 hi=s.rolling(20,min_periods=15).max(); dd=s/hi-1
 age=[]
 for i in range(len(s)):
  w=s.iloc[max(0,i-19):i+1]
  age.append((len(w)-1)-np.nanargmax(w.values) if w.notna().sum()>=15 else np.nan)
 return dd/pd.Series(age,index=s.index).replace(0,1)
f=pd.DataFrame({a:dd_persist(p[a]) for a in A})
fw={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
v=pd.DataFrame({a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut,'volume'].astype(float).replace(0,np.nan) for a in A})
rv=np.log(v/v.rolling(20,min_periods=15).mean())
def asym(x):
 out=[]
 for e in range(len(x)):
  w=x.iloc[max(0,e-59):e+1]; m=med.reindex(w.index); lo=m<0; hi=m>=0
  out.append(w[lo].corr(m[lo])-w[hi].corr(m[hi]) if lo.sum()>=12 and hi.sum()>=12 else np.nan)
 return pd.Series(out,index=x.index)
lib={'risk_adjusted_trend_20d':trend,'relative_volume_participation_20d':rv,'volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'realized_volatility_20obs':r.rolling(20,min_periods=15).std(),'cross_asset_beta_compression_20obs':pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(med) for a in A}),'risk_adjusted_trend_acceleration_20_60d':trend-(p/p.shift(60)-1)/r.rolling(60,min_periods=45).std(),'correlation_asymmetry_60obs':pd.DataFrame({a:asym(r[a]) for a in A})}
print('FACTOR drawdown_persistence_20obs = (close/20obs_rolling_max-1)/max(days_since_20obs_peak,1)')
print('visible_through',cut.date(),'source_start',p.index.min().date(),'assets',len(A))
def test(h):
 out=[]; cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),fw[h].loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: out.append((dt,z.x.corr(z.y,method='spearman'))); cov.append(len(z)/15)
 return pd.Series(dict(out)),np.mean(cov) if cov else np.nan
for h in [1,5,10,20]:
 x,c=test(h); sd=x.std(ddof=1)
 print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for n,mask in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
  q=x[mask]; print(f' REGIME H={h} {n} n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(tos):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=0
for n,g in lib.items():
 z=pd.concat([f.stack().rename('new'),g.stack().rename('old')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); rho=z.new.corr(z.old,method='spearman'); mx=max(mx,abs(rho)); print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; admitted_library_count={len(glob.glob("factors/*.json"))}')
