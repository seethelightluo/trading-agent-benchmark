"""miner_1: trailing return-autocorrelation resilience factor, evaluated without future inputs."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2026-12-30'); H=[1,5,10,20]
C={}; R={}; V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date');d=d.loc[:cutoff]
 C[a]=d.close.replace(0,np.nan);R[a]=C[a].pct_change();V[a]=d.volume.replace(0,np.nan)
# High score is a more negative lag-one autocorrelation: short-horizon shocks have tended to mean-revert.
def ac1(x):
 x=np.asarray(x); x=x[np.isfinite(x)]
 return np.nan if len(x)<15 or np.std(x)==0 else -np.corrcoef(x[:-1],x[1:])[0,1]
F=pd.DataFrame({a:R[a].rolling(20,min_periods=15).apply(ac1,raw=True) for a in A})
Y={h:pd.DataFrame({a:C[a].shift(-h)/C[a]-1 for a in A}) for h in H}
def stats(h,span=None):
 f=F if span is None else F.loc[span[0]:span[1]];y=Y[h] if span is None else Y[h].loc[span[0]:span[1]];z=[];nn=[]
 for d in f.index:
  q=pd.concat([f.loc[d],y.loc[d]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   x=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(x):z.append(x);nn.append(len(q))
 z=np.array(z);return len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),np.mean(nn),np.min(nn)
print('FACTOR lag1_return_autocorrelation_reversal_20 CUTOFF',cutoff.date())
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'COVERAGE',float(F.notna().stack().mean()))
for h in H: print('H',h,'DATES IC ICIR HIT MEAN_NAMES MIN_NAMES',stats(h))
for lab,sp in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-30'))]:print('REGIME10',lab,stats(10,sp))
print('TURNOVER',float(F.rank(axis=1,pct=True).diff().abs().stack().mean()))
# Reconstruct all admitted signals and require empirical evidence against every one.
def qeff(a): return (C[a].pct_change(20).abs()/R[a].abs().rolling(20,min_periods=15).sum())*(1-R[a].rolling(20,min_periods=15).std().rolling(60,min_periods=40).apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1],raw=False))
base={'relative_volume_participation':pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}),'risk_adjusted_trend':pd.DataFrame({a:(C[a]/C[a].shift(20)-1)/R[a].rolling(20,min_periods=15).std() for a in A}),'volnorm_reversal_5':pd.DataFrame({a:-(C[a]/C[a].shift(5)-1)/R[a].rolling(5,min_periods=4).std() for a in A}),'volscaled_reversal_1':pd.DataFrame({a:-R[a]/R[a].rolling(20,min_periods=15).std() for a in A}),'quiet_trend_path_efficiency':pd.DataFrame({a:qeff(a) for a in A})}
# VIX regime signal
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date').close.replace(0,np.nan).loc[:cutoff]; sg=np.where(v/v.shift(20)-1>0,-1.,1.);base['vix_conditioned_risk_adjusted_trend']=pd.DataFrame({a:sg*((C[a]/C[a].shift(20)-1)/R[a].rolling(20,min_periods=15).std()) for a in A})
mx=0;who=''
for n,x in base.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=spearmanr(z.f,z.x).statistic;print('LIBRARY',n,'RHO',rho,'CELLS',len(z));
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who)
