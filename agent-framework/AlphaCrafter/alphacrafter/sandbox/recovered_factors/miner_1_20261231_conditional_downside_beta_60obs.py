"""miner_1: validate conditional downside-beta ratio, 60 observations."""
import glob
import numpy as np
import pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Runtime research date 2026-12-31: never use any future row; latest completed day is 2026-12-30.
END=pd.Timestamp('2026-12-30')
def load(a):
 return pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
P={a:load(a).close.astype(float) for a in ASSETS}; V={a:load(a).volume.astype(float) for a in ASSETS}
rr=pd.DataFrame({a:P[a].pct_change() for a in ASSETS}); med=rr.median(axis=1)
# Difference between beta to the broad cross-asset median on down-median and up-median days.
# Positive means the asset has disproportionate systematic exposure specifically during selloffs.
def dbeta(x,m):
 z=pd.concat([x,m],axis=1).dropna(); dn=z[z.iloc[:,1]<0]; up=z[z.iloc[:,1]>=0]
 if len(dn)<12 or len(up)<12 or dn.iloc[:,1].var()==0 or up.iloc[:,1].var()==0:return np.nan
 return dn.iloc[:,0].cov(dn.iloc[:,1])/dn.iloc[:,1].var()-up.iloc[:,0].cov(up.iloc[:,1])/up.iloc[:,1].var()
f=pd.DataFrame({a:[dbeta(rr[a].loc[:dt].tail(60),med.loc[:dt].tail(60)) for dt in rr.index] for a in ASSETS},index=rr.index)
fw={h:pd.DataFrame({a:P[a].shift(-h)/P[a]-1 for a in ASSETS}) for h in (1,5,10,20)}
lib={
'risk_adjusted_trend':pd.DataFrame({a:(P[a]/P[a].shift(20)-1)/rr[a].rolling(20,min_periods=15).std() for a in ASSETS}),
'relative_volume':pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in ASSETS}),
'volnorm_reversal':pd.DataFrame({a:-(P[a]/P[a].shift(5)-1)/rr[a].rolling(5,min_periods=4).std() for a in ASSETS}),
'realized_volatility':rr.rolling(20,min_periods=15).std(),
'beta_compression':pd.DataFrame({a:rr[a].rolling(20,min_periods=15).corr(med) for a in ASSETS})}
lib['trend_acceleration']=lib['risk_adjusted_trend']-pd.DataFrame({a:(P[a]/P[a].shift(60)-1)/rr[a].rolling(60,min_periods=45).std() for a in ASSETS})
asym={}
for a in ASSETS:
 vals={}
 for dt in rr.index:
  z=pd.concat([rr[a],med],axis=1).loc[:dt].tail(60).dropna();dn=z[z.iloc[:,1]<0];up=z[z.iloc[:,1]>=0]
  vals[dt]=dn.iloc[:,0].corr(dn.iloc[:,1])-up.iloc[:,0].corr(up.iloc[:,1]) if len(dn)>=10 and len(up)>=10 else np.nan
 asym[a]=pd.Series(vals)
lib['correlation_asymmetry']=pd.DataFrame(asym)
print('FACTOR conditional_downside_beta_60obs = beta(asset, median cross-asset return | median<0) - beta(asset, median | median>=0), trailing 60 observations')
print('visible through',END.date(),'assets',len(ASSETS),'source dates',f.index.min().date(),f.index.max().date())
for h,y in fw.items():
 obs=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:obs.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(obs));sd=x.std(ddof=1)
 print(f'H={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cov):.4f}')
 for name,mask in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
  q=x[mask]; print(f' {name}: n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
r=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1
for n,old in lib.items():
 z=pd.concat([f.stack().rename('new'),old.stack().rename('old')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.new.corr(z.old,method='spearman')
 print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);closest=n
print(f'max_abs_library_correlation={mx:.6f}; closest={closest}; library_json_count={len(glob.glob("factors/*.json"))}')
