"""miner_1: validate 20-day return sign-balance factor, using only data visible through 2027-04-07."""
import numpy as np, pandas as pd, glob
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-04-07')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 return d.close.astype(float),d.volume.astype(float).replace(0,np.nan)
p={};v={}
for a in AS:p[a],v[a]=load(a)
def frame(fn, src=p): return pd.DataFrame({a:fn(x) for a,x in src.items()})
r=frame(lambda x:x.pct_change()); med=r.median(axis=1)
# A high score means a large majority of the last 20 native returns were positive.
# It separates breadth/persistence of the path from magnitude-based trend.
f=r.gt(0).astype(float).rolling(20,min_periods=15).mean()-0.5
fw={h:pd.DataFrame({a:x.shift(-h)/x-1 for a,x in p.items()}) for h in (1,5,10,20)}
vol=frame(lambda x:x.pct_change().rolling(20,min_periods=15).std())
fast=frame(lambda x:(x/x.shift(20)-1)/x.pct_change().rolling(20,min_periods=15).std())
slow=frame(lambda x:(x/x.shift(60)-1)/x.pct_change().rolling(60,min_periods=45).std())
rev=frame(lambda x:-(x/x.shift(5)-1)/x.pct_change().rolling(5,min_periods=4).std())
confirm=frame(lambda x:np.log(x/x.rolling(20,min_periods=15).mean()),v)
beta=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(med) for a in AS})
lib={'miner_3_risk_adjusted_trend_20d':fast,'miner_3_relative_volume_participation_20d':confirm,'miner_1_volnorm_reversal_5obs':rev,'miner_2_realized_volatility_20obs':vol,'cross_asset_beta_compression_20obs':beta,'miner_3_risk_adjusted_trend_acceleration_20_60d':fast-slow,'miner_2_return_skewness_20obs':r.rolling(20,min_periods=15).skew()}
asym={}
for a in AS:
 vals=[]
 for dt in r.index:
  z=pd.concat([r[a],med],axis=1).loc[:dt].tail(60).dropna(); dn=z[z.iloc[:,1]<0]; up=z[z.iloc[:,1]>=0]
  vals.append(dn.iloc[:,0].corr(dn.iloc[:,1])-up.iloc[:,0].corr(up.iloc[:,1]) if len(dn)>=10 and len(up)>=10 else np.nan)
 asym[a]=vals
lib['miner_1_correlation_asymmetry_60obs']=pd.DataFrame(asym,index=r.index)
lib['miner_3_return_persistence_autocorr_20obs']=r.rolling(20,min_periods=15).apply(lambda x: pd.Series(x).autocorr(),raw=False)
print('FACTOR return_sign_balance_20obs = rolling mean(1[daily native return > 0], 20 observations) - 0.5; high means broad positive return participation, independent of return magnitude')
print('validation_date=2027-04-08 visible_through=',END.date(),'assets=',len(AS),'period=',r.index.min().date(),'to',END.date())
ics={}
for h,y in fw.items():
 vals=[]; cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(vals));ics[h]=x; sd=x.std(ddof=1)
 print(f'H={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} mean_date_coverage={np.mean(cov):.4f}')
 for name,mask in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-current',x.index>='2025-01-01')]:
  q=x[mask]; print(f' {name}: n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(tos):.6f}; raw_coverage={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1;closest=''
for n,o in lib.items():
 z=pd.concat([f.stack().rename('f'),o.stack().rename('o')],axis=1).dropna(); rho=z.f.corr(z.o,method='spearman')
 print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);closest=n
print(f'max_abs_library_correlation={mx:.6f}; closest={closest}; active_library_json_count={len([x for x in glob.glob("factors/*.json") if not x.endswith(".bak")])}')
