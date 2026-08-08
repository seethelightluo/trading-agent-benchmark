"""miner_3: validate one factor -- 20/60 observation trend acceleration."""
import pandas as pd, numpy as np, glob
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
H=[1,5,10,20]
F={}; FW={}; LIB={k:{} for k in ['risk_adjusted_trend_20d','ravmom_20obs','volnorm_reversal_5obs','relative_volume_participation_20d','realized_volatility_20obs']}
for a in ASSETS:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 p=d.close.astype(float); r=p.pct_change(); v=d.volume.astype(float).replace(0,np.nan)
 # Acceleration: recent 20-observation return less preceding 40-observation return, normalized by current risk.
 F[a]=(p/p.shift(20)-p.shift(20)/p.shift(60))/r.rolling(20,min_periods=15).std()
 for h in H: FW.setdefault(h,{})[a]=p.shift(-h)/p-1
 LIB['risk_adjusted_trend_20d'][a]=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
 LIB['ravmom_20obs'][a]=LIB['risk_adjusted_trend_20d'][a]
 LIB['volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
 LIB['relative_volume_participation_20d'][a]=np.log(v/v.rolling(20,min_periods=15).mean())
 LIB['realized_volatility_20obs'][a]=r.rolling(20,min_periods=15).std()
f=pd.DataFrame(F).sort_index(); lib={k:pd.DataFrame(x).sort_index() for k,x in LIB.items()}
print('FACTOR trend_acceleration_20_60obs = [close_t/close_(t-20) - close_(t-20)/close_(t-60)] / rolling_std(return,20); higher = accelerating risk-adjusted trend')
print('history',f.index.min().date(),f.index.max().date(),'assets',len(ASSETS))
def getic(h):
 fw=pd.DataFrame(FW[h]); o=[]; cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: o.append((dt,z.x.corr(z.y,method='spearman')));cov.append(len(z)/15)
 return pd.Series(dict(o)),float(np.mean(cov))
for h in H:
 x,c=getic(h); ir=x.mean()/x.std(ddof=1)
 print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={ir:.6f} hit={(x>0).mean():.4f} IC_se={x.std(ddof=1)/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for n,m in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  y=x[m]; print(f'  {n}: n={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
r=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'mean_rank_turnover={np.mean(turn):.6f}; signal_cells={f.notna().sum().sum()}/{f.size} ({f.notna().mean().mean():.4f})')
mx=0
for n,l in lib.items():
 z=pd.concat([f.stack().rename('x'),l.stack().rename('y')],axis=1).dropna(); rho=z.x.corr(z.y,method='spearman');mx=max(mx,abs(rho));print(f'library_{n}_rho={rho:.6f}; cells={len(z)}')
print(f'library_files={len(glob.glob("factors/*.json"))}; max_abs_library_correlation={mx:.6f}')
