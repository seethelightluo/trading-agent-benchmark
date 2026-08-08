"""miner_1: validate return-volume elasticity: sensitivity of absolute moves to relative participation."""
import glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; P={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
 P[a]=d.close.astype(float); V[a]=d.volume.astype(float)
p=pd.DataFrame(P).sort_index(); v=pd.DataFrame(V).sort_index(); r=p.pct_change()
# One interpretable idea: 20-observation correlation of absolute daily return with log relative volume.
# Positive values mean participation reliably expands on large moves (elastic/liquid discovery);
# negative values mean large moves occur despite muted participation (fragile price formation).
rv=np.log(v/v.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
f=pd.DataFrame({a:r[a].abs().rolling(20,min_periods=15).corr(rv[a]) for a in A})
fw={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
lib={
 'miner_3_risk_adjusted_trend_20d':(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std(),
 'miner_3_relative_volume_participation_20d':rv,
 'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),
 'miner_2_realized_volatility_20obs':r.rolling(20,min_periods=15).std(),
 'cross_asset_beta_compression_20obs':pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(r.median(axis=1)) for a in A}),
 'risk_adjusted_trend_acceleration_20_60d':(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()-(p/p.shift(60)-1)/r.rolling(60,min_periods=45).std()}
med=r.median(axis=1)
def asym(x):
 ans=[]
 for end in range(len(x)):
  w=x.iloc[max(0,end-59):end+1]; m=med.reindex(w.index); lo=m<0; hi=m>=0
  ans.append(w[lo].corr(m[lo])-w[hi].corr(m[hi]) if lo.sum()>=12 and hi.sum()>=12 else np.nan)
 return pd.Series(ans,index=x.index)
lib['miner_1_correlation_asymmetry_60obs']=pd.DataFrame({a:asym(r[a]) for a in A})
print('FACTOR return_volume_elasticity_20obs = corr_20(abs(return), log(volume/mean_20(volume)))')
print('visible_history',f.index.min().date(),f.index.max().date(),'assets',len(A))
def getic(h):
 vals=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),fw[h].loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: vals.append((dt,z.x.corr(z.y,method='spearman'))); cov.append(len(z)/len(A))
 return pd.Series(dict(vals)),np.mean(cov)
for h in [1,5,10,20]:
 x,c=getic(h); sd=x.std(ddof=1); print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 if h==10:
  for nm,mask in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
   y=x[mask]; print(f' REGIME {nm} n={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(turns):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=0
for n,g in lib.items():
 z=pd.concat([f.stack().rename('a'),g.stack().rename('b')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); rho=z.a.corr(z.b,method='spearman'); mx=max(mx,abs(rho)); print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; admitted_library_count={len(glob.glob("factors/*.json"))}')
