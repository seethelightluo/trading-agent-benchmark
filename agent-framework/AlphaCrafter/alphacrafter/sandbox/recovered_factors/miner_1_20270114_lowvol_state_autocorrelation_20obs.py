"""miner_1 research: low-volatility-state 1-day return autocorrelation (20 observations)."""
import glob,numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2027-01-13');P={};V={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cutoff];P[a]=d.close.astype(float);V[a]=d.volume.astype(float).replace(0,np.nan)
p=pd.DataFrame(P);v=pd.DataFrame(V);r=p.pct_change(); vol5=r.rolling(5,min_periods=4).std(); state=vol5.le(vol5.rolling(20,min_periods=15).median())
def state_ac(x,st):
 out=[]
 for i in range(len(x)):
  z=pd.concat([x.shift(1).iloc[max(0,i-19):i+1],x.iloc[max(0,i-19):i+1],st.iloc[max(0,i-19):i+1]],axis=1).dropna();z=z[z.iloc[:,2]]
  out.append(z.iloc[:,0].corr(z.iloc[:,1]) if len(z)>=8 else np.nan)
 return pd.Series(out,index=x.index)
f=pd.DataFrame({a:state_ac(r[a],state[a]) for a in A}); fw={h:p.shift(-h)/p-1 for h in [1,5,10,20]};rv=np.log(v/v.rolling(20,min_periods=15).mean())
med=r.median(axis=1)
def asym(x):
 out=[]
 for i in range(len(x)):
  w=x.iloc[max(0,i-59):i+1];m=med.reindex(w.index);lo=m<0;hi=m>=0;out.append(w[lo].corr(m[lo])-w[hi].corr(m[hi]) if lo.sum()>=12 and hi.sum()>=12 else np.nan)
 return pd.Series(out,index=x.index)
lib={'risk_adjusted_trend':(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std(),'relative_volume':rv,'volnorm_reversal':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'realized_vol':r.rolling(20,min_periods=15).std(),'beta_compression':pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(med) for a in A}),'trend_acceleration':(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()-(p/p.shift(60)-1)/r.rolling(60,min_periods=45).std(),'correlation_asymmetry':pd.DataFrame({a:asym(r[a]) for a in A})}
print('FACTOR lowvol_state_autocorrelation_20obs = corr(return[t-1],return[t]) on trailing-20 days where 5d vol <= own trailing-20 median');print('visible',p.index.min().date(),p.index.max().date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 o=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),fw[h].loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:o.append((dt,z.x.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(o));ics[h]=x;s=x.std(ddof=1);print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/s:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cov):.4f}')
 if h==5:
  for n,q in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
   y=x[q];print(f' REGIME {n} n={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(tos):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=0
for n,g in lib.items():
 z=pd.concat([f.stack().rename('a'),g.stack().rename('b')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.a.corr(z.b,method='spearman');mx=max(mx,abs(rho));print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; admitted_library_count={len(glob.glob("factors/*.json"))}')
