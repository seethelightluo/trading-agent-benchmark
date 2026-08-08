"""miner_1: dispersion-conditioned drawdown duration (one factor idea).
Signal is negative time spent below a rolling 20-observation high, amplified only
when cross-asset 5-observation return dispersion is elevated. This distinguishes
persistent weakness in differentiated markets from short-lived broad corrections.
"""
import glob, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-02-10'); P={}; V={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d.close.astype(float); V[a]=d.volume.astype(float)
p=pd.DataFrame(P);v=pd.DataFrame(V);r=p.pct_change();med=r.median(axis=1)
# Consecutive native observations below the trailing high (0 at a new 20d high).
peak=p.rolling(20,min_periods=15).max(); below=(p < peak*(1-1e-12)).astype(float)
dur=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
 run=0; out=[]
 for x,ok in zip(below[a],peak[a].notna()):
  run=(run+1 if x else 0) if ok else 0
  out.append(run if ok else np.nan)
 dur[a]=out
# A common dispersion-state multiplier does not alter same-day ranks, but is an
# interpretable conditional activation rule; zero signal in low dispersion states.
disp=(p/p.shift(5)-1).std(axis=1)
state=(disp > disp.rolling(60,min_periods=40).median()).astype(float)
f=-(dur/20).mul(state,axis=0).where(peak.notna())
fw={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
def asym(x):
 q=[]
 for i in range(len(x)):
  w=x.iloc[max(0,i-59):i+1];m=med.reindex(w.index);lo=m<0;hi=m>=0
  q.append(w[lo].corr(m[lo])-w[hi].corr(m[hi]) if lo.sum()>=12 and hi.sum()>=12 else np.nan)
 return pd.Series(q,index=x.index)
lib={'risk_adjusted_trend':(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std(),'relative_volume':np.log(v/v.rolling(20,min_periods=15).mean()),'volnorm_reversal':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'realized_vol':r.rolling(20,min_periods=15).std(),'beta_compression':pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(med) for a in A}),'trend_acceleration':(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()-(p/p.shift(60)-1)/r.rolling(60,min_periods=45).std(),'correlation_asymmetry':pd.DataFrame({a:asym(r[a]) for a in A}),'return_skewness':r.rolling(20,min_periods=15).skew()}
print('FACTOR dispersion_conditioned_drawdown_duration_20obs = -consecutive_below_20d_high/20 if cross_asset_5d_dispersion > trailing_60d_median else 0')
print('visible',p.index.min().date(),p.index.max().date(),'assets',len(A))
for h in [1,5,10,20]:
 o=[];c=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),fw[h].loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.x.nunique()>1:o.append((dt,z.x.corr(z.y,method='spearman')));c.append(len(z)/15)
 x=pd.Series(dict(o)); sd=x.std(ddof=1)
 print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} coverage={np.mean(c):.4f}')
 if h==5:
  for n,m in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-current',x.index>='2025-01-01')]:
   y=x[m];print(f' REGIME {n} n={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=0
for n,g in lib.items():
 z=pd.concat([f.stack().rename('a'),g.stack().rename('b')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.a.corr(z.b,method='spearman');mx=max(mx,abs(rho));print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}; admitted_library_count={len(glob.glob("factors/*.json"))}')
