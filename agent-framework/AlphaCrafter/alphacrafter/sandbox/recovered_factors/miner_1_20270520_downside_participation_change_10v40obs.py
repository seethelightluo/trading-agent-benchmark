"""miner_1: validate downside-participation change, one conditional path-behaviour idea."""
import numpy as np, pandas as pd
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-05-19')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 return d.close.astype(float),d.volume.astype(float).replace(0,np.nan)
p={};v={}
for a in AS:p[a],v[a]=load(a)
def nf(fun,src=p):return pd.DataFrame({a:fun(x.dropna()).reindex(x.index) for a,x in src.items()})
r=nf(lambda x:x.pct_change()); med=r.median(axis=1)
# High: recent downside participation is lower than its own 40-observation baseline,
# i.e. asset has begun to decouple from broad cross-asset losses.
def dpi(x):
 z=x.pct_change(); m=med.reindex(z.index); cap=z.where(m<0).rolling(10,min_periods=6).mean()/m.where(m<0).rolling(10,min_periods=6).mean()
 return -(cap-cap.rolling(40,min_periods=25).mean())/cap.rolling(40,min_periods=25).std()
f=nf(dpi)
fw={h:pd.DataFrame({a:x.pct_change(h).shift(-h) for a,x in p.items()}) for h in(1,5,10,20)}
vol=nf(lambda x:x.pct_change().rolling(20,min_periods=15).std()); fast=nf(lambda x:(x/x.shift(20)-1)/x.pct_change().rolling(20,min_periods=15).std());slow=nf(lambda x:(x/x.shift(60)-1)/x.pct_change().rolling(60,min_periods=45).std())
lib={'miner_3_risk_adjusted_trend_20d':fast,'miner_3_relative_volume_participation_20d':nf(lambda x:np.log(x/x.rolling(20,min_periods=15).mean()),v),'miner_1_volnorm_reversal_5obs':nf(lambda x:-(x/x.shift(5)-1)/x.pct_change().rolling(5,min_periods=4).std()),'miner_2_realized_volatility_20obs':vol,'miner_3_risk_adjusted_trend_acceleration_20_60d':fast-slow,'miner_3_return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).apply(lambda z:z.autocorr(1) if len(z)>=15 else np.nan,raw=False),'miner_1_return_sign_balance_20obs':nf(lambda x:x.pct_change().gt(0).rolling(20,min_periods=15).mean()),'miner_3_return_directional_efficiency_20obs':nf(lambda x:x.pct_change().rolling(20,min_periods=15).sum().abs()/x.pct_change().abs().rolling(20,min_periods=15).sum()),'miner_2_cross_asset_beta_compression_20obs':pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(med) for a in AS})}
as={}
for a in AS:
 q=[]
 for dt in r.index:
  z=pd.concat([r[a],med],axis=1).loc[:dt].tail(60).dropna();dn=z[z.iloc[:,1]<0];up=z[z.iloc[:,1]>=0];q.append(dn.iloc[:,0].corr(dn.iloc[:,1])-up.iloc[:,0].corr(up.iloc[:,1]) if len(dn)>=10 and len(up)>=10 else np.nan)
 as[a]=q
lib['miner_1_correlation_asymmetry_60obs']=pd.DataFrame(as,index=r.index)
print('FACTOR downside_participation_change_10v40obs: negative z-score of recent 10-down-session downside-capture relative to trailing 40-observation baseline')
print('visible_through',END.date(),'assets',len(AS))
for h,y in fw.items():
 vals=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(vals));sd=x.std(ddof=1);print(f'H={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cov):.4f}')
 for nm,ma in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-current',x.index>='2025-01-01')]:
  q=x[ma];print(f' {nm}: n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1;closest=''
for n,o in lib.items():
 z=pd.concat([f.stack().rename('f'),o.stack().rename('o')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.f.corr(z.o,method='spearman');print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);closest=n
print(f'max_abs_library_correlation={mx:.6f}; closest={closest}; library_count={len(lib)}')
