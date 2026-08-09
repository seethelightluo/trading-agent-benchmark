"""miner_1: relative-volume shock reversal, validated without look-ahead."""
import glob, json, numpy as np, pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2027-03-24')
def rd(a): return pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
p={}; r={}; f={}; fw={}; lib={}
for a in ASSETS:
 d=rd(a); p[a]=d.close.astype(float); r[a]=p[a].pct_change(); v=d.volume.astype(float)
 # High score: a negative one-day return with an unusually large volume surprise; expected short-horizon exhaustion reversal.
 lv=np.log(v/v.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
 f[a]=-r[a]*lv/r[a].rolling(20,min_periods=15).std()
 fw[a]={h:p[a].shift(-h)/p[a]-1 for h in (1,5,10,20)}
F=pd.DataFrame(f); R=pd.DataFrame(r); med=R.median(axis=1)
# Reconstruct every admitted factor from its persisted definition for required signal-correlation evidence.
for a in ASSETS:
 ret=R[a]; fast=(p[a]/p[a].shift(20)-1)/ret.rolling(20,min_periods=15).std(); slow=(p[a]/p[a].shift(60)-1)/ret.rolling(60,min_periods=45).std()
 v=rd(a).volume.astype(float); rel=np.log(v/v.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
 lib.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=fast
 lib.setdefault('miner_3_relative_volume_participation_20d',{})[a]=rel
 lib.setdefault('miner_1_volnorm_reversal_5obs',{})[a]=-(p[a]/p[a].shift(5)-1)/ret.rolling(5,min_periods=4).std()
 lib.setdefault('miner_2_realized_volatility_20obs',{})[a]=ret.rolling(20,min_periods=15).std()
 lib.setdefault('risk_adjusted_trend_acceleration_20_60d',{})[a]=fast-slow
for a in ASSETS: lib.setdefault('cross_asset_beta_compression_20obs',{})[a]=R[a].rolling(20,min_periods=15).corr(med)
for a in ASSETS:
 x=[]
 for i in range(len(R)):
  xx=R[a].iloc[max(0,i-59):i+1]; yy=med.iloc[max(0,i-59):i+1]; dn=yy<0; up=yy>=0
  x.append(xx[dn].corr(yy[dn])-xx[up].corr(yy[up]) if dn.sum()>=12 and up.sum()>=12 else np.nan)
 lib.setdefault('miner_1_correlation_asymmetry_60obs',{})[a]=pd.Series(x,index=R.index)
# Return skewness definition is rolling standardized third moment.
for a in ASSETS: lib.setdefault('miner_2_return_skewness_20obs',{})[a]=R[a].rolling(20,min_periods=15).skew()
print('FACTOR relative_volume_shock_reversal_1_20obs = -return_1d*log(volume/mean(volume,20))/std(return,20)')
print('visible history',F.index.min().date(),F.index.max().date(),'instruments',len(ASSETS))
def ics(h):
 Y=pd.DataFrame({a:fw[a][h] for a in ASSETS}); z=[]; cov=[]
 for dt in F.index:
  q=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8: z.append((dt,q.f.corr(q.y,method='spearman')));cov.append(len(q)/15)
 return pd.Series(dict(z)),np.mean(cov)
for h in (1,5,10,20):
 x,c=ics(h); s=x.std(ddof=1); print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/s:.6f} hit={(x>0).mean():.4f} coverage={c:.4f}')
 for nm,m in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-current',x.index>='2025-01-01')]:
  q=x[m];print(f' {nm} n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8: turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(turns):.6f}; cells={F.notna().sum().sum()}/{F.size}={F.notna().mean().mean():.4f}')
mx=0
for nm,d in lib.items():
 q=pd.concat([F.stack().rename('new'),pd.DataFrame(d).stack().rename('old')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); rho=q.new.corr(q.old,method='spearman'); mx=max(mx,abs(rho));print(f'LIB {nm} rho={rho:.6f} cells={len(q)}')
print(f'max_abs_library_correlation={mx:.6f}; admitted_json_count={len(glob.glob("factors/*.json"))}')
