"""miner_1: validate up/down magnitude asymmetry over 20 native observations."""
import numpy as np,pandas as pd,glob
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-04-21')
def ld(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 return d.close.astype(float),d.volume.astype(float).replace(0,np.nan)
p={};v={}
for a in AS:p[a],v[a]=ld(a)
def native(fun):return pd.DataFrame({a:fun(x.dropna()).reindex(x.index) for a,x in p.items()})
r=native(lambda x:x.pct_change()); market=r.median(axis=1)
# Difference of average positive and negative move magnitudes, normalized by total average magnitude.
def asym(x):
 q=x.pct_change(); up=q.where(q>0).rolling(20,min_periods=15).mean(); dn=(-q.where(q<0)).rolling(20,min_periods=15).mean()
 return (up-dn)/(up+dn+1e-12)
f=native(asym)
fw={h:pd.DataFrame({a:x.dropna().shift(-h)/x.dropna()-1 for a,x in p.items()}) for h in [1,5,10,20]}
vol=native(lambda x:x.pct_change().rolling(20,min_periods=15).std()); fast=native(lambda x:(x/x.shift(20)-1)/x.pct_change().rolling(20,min_periods=15).std()); slow=native(lambda x:(x/x.shift(60)-1)/x.pct_change().rolling(60,min_periods=45).std())
lib={
'miner_3_risk_adjusted_trend_20d':fast,
'miner_3_relative_volume_participation_20d':pd.DataFrame({a:np.log(x.dropna()/x.dropna().rolling(20,min_periods=15).mean()) for a,x in v.items()}),
'miner_1_volnorm_reversal_5obs':native(lambda x:-(x/x.shift(5)-1)/x.pct_change().rolling(5,min_periods=4).std()),
'miner_2_realized_volatility_20obs':vol,
'miner_3_risk_adjusted_trend_acceleration_20_60d':fast-slow,
'miner_2_return_skewness_20obs':r.rolling(20,min_periods=15).skew(),
'miner_1_return_sign_balance_20obs':native(lambda x:(x.pct_change()>0).astype(float).rolling(20,min_periods=15).mean()-.5),
'miner_3_return_directional_efficiency_20obs':native(lambda x:x.pct_change().rolling(20,min_periods=15).sum().abs()/x.pct_change().abs().rolling(20,min_periods=15).sum()),
'miner_3_return_persistence_autocorr_20obs':native(lambda x:x.pct_change().rolling(20,min_periods=15).apply(lambda z:pd.Series(z).autocorr(1),raw=False))}
# correlation asymmetry: beta on market-down minus beta on market-up
ca={}
for a in AS:
 o=[]
 for dt in r.index:
  z=pd.concat([r[a],market],axis=1).loc[:dt].tail(60).dropna();dn=z[z.iloc[:,1]<0];up=z[z.iloc[:,1]>=0]
  o.append(dn.iloc[:,0].corr(dn.iloc[:,1])-up.iloc[:,0].corr(up.iloc[:,1]) if len(dn)>=10 and len(up)>=10 else np.nan)
 ca[a]=o
lib['miner_1_correlation_asymmetry_60obs']=pd.DataFrame(ca,index=r.index)
print('FACTOR up_down_magnitude_asymmetry_20obs = (mean positive daily return - mean absolute negative daily return)/(sum), trailing 20 native observations, min 15; higher means upside moves dominate in magnitude.')
print('visible through',END.date(),'assets',len(AS),'library factors',len(lib))
allstats={}
for h,y in fw.items():
 vals=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: vals.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(vals)); sd=x.std(ddof=1); allstats[h]=x
 print(f'H={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} mean_coverage={np.mean(cov):.4f}')
for label,mask in [('2020',allstats[10].index<'2021-01-01'),('2021-22',(allstats[10].index>='2021-01-01')&(allstats[10].index<'2023-01-01')),('2023-24',(allstats[10].index>='2023-01-01')&(allstats[10].index<'2025-01-01')),('2025-current',allstats[10].index>='2025-01-01')]:
 x=allstats[10][mask];print(f'REGIME H10 {label} n={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);t=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(t):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1;closest=''
for n,o in lib.items():
 z=pd.concat([f.stack().rename('f'),o.stack().rename('o')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.f.corr(z.o,method='spearman');print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);closest=n
print(f'max_abs_library_correlation={mx:.6f}; closest={closest}; json_count={len(glob.glob("factors/*.json"))}')
