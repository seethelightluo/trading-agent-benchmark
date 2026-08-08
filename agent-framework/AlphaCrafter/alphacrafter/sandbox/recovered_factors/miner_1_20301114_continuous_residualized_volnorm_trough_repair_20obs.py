"""Single idea: continuous residualized volatility-normalized drawdown repair speed."""
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2030-11-13')
def ld(a,c='close',idx=False):
 d='../persistent/index_data/' if idx else '../persistent/stock_data/'
 return pd.read_csv(d+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,c].astype(float)
def beta(x,y,w=60,cond=None):
 if isinstance(x,pd.DataFrame): return pd.DataFrame({a:beta(x[a],y,w,cond[a] if isinstance(cond,pd.DataFrame) else cond) for a in x})
 yy=y.reindex(x.index,method='ffill'); cc=cond.reindex(x.index).fillna(False) if cond is not None else None;xx=x.where(cc) if cc is not None else x;yy=yy.where(cc) if cc is not None else yy
 return xx.rolling(w,min_periods=12).cov(yy).div(yy.rolling(w,min_periods=12).var())
p=pd.DataFrame({a:ld(a) for a in A});r=p.pct_change(fill_method=None);hi=pd.DataFrame({a:ld(a,'high') for a in A});lo=pd.DataFrame({a:ld(a,'low') for a in A});V=pd.DataFrame({a:ld(a,'volume') for a in A})
m=r.median(axis=1);stress=(-m.shift()/m.rolling(60,min_periods=30).std()).clip(-4,4); vol20=r.rolling(20,min_periods=15).std();trough=p.rolling(20,min_periods=15).min()
# Continuous rather than conditional: five-day repair from a rolling trough, scaled by own volatility; residual removes trend, directional breadth and vol.
raw=np.log(p/trough.shift(5)).div(vol20*np.sqrt(5));trend=(p/p.shift(20)-1).div(vol20);balance=(r>0).rolling(20,min_periods=15).mean()-.5
f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in p.index:
 q=pd.concat([raw.loc[t],trend.loc[t],balance.loc[t],vol20.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8:
  X=np.column_stack([np.ones(len(q)),q.iloc[:,1:].values]);f.loc[t,q.index]=q.iloc[:,0].values-X@np.linalg.lstsq(X,q.iloc[:,0].values,rcond=None)[0]
print('CANDIDATE continuous_residualized_volnorm_trough_repair_20obs cutoff',END.date(),'assets',len(A))
print('expression=cross_sectional_residual[log(close[t]/rolling_min(close,20)[t-5])/(rolling_std(return,20)[t]*sqrt(5)) | risk_adjusted_trend_20d, return_sign_balance_20obs, realized_volatility_20obs]')
allx={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p).sub(1);z=[];cv=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:z.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));cv.append(len(q)/15)
 x=pd.Series(dict(z));allx[h]=x;ic=x.mean();ir=ic/x.std(ddof=1)
 print(f'H={h} dates={len(x)} IC={ic:.6f} ICIR={ir:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cv):.4f} mean_instruments={15*np.mean(cv):.2f} GATE={abs(ic)>=.007 and abs(ir)>=.084}')
h=max(allx,key=lambda h:abs(allx[h].mean()*(allx[h].mean()/allx[h].std(ddof=1)));x=allx[h];print('BEST_HORIZON',h)
for n,l,u in [('2020-21','2020','2022'),('2022-23','2022','2024'),('2024-25','2024','2026'),('2026-current','2026','2031')]:
 z=x[(x.index>=l)&(x.index<u)];print(f'REGIME {n} dates={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}; concentration_mean_sd={f.std(axis=1).mean():.6f}')
# Use all existing candidate signal artifacts as exact contemporaneous library evidence where present; plus reconstruct primary dependencies.
paths={'residualized_volnorm_drawdown_repair_speed_20_60obs':'scripts/miner_1_20301017_residualized_volnorm_drawdown_repair_speed_20_60obs_candidate_signal.pkl','volume_confirmed_repair':'scripts/miner_1_20301031_residualized_volume_confirmed_drawdown_repair_20_60obs_candidate_signal.pkl','continuous_dispersion_residual':'scripts/miner_3_continuous_dispersion_candidate_signal.pkl','common_stress_volume':'scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs_candidate_signal.pkl'}
lib={}
for n,z in paths.items():
 try:lib[n]=pd.read_pickle(z).loc[:END]
 except:pass
lib['risk_adjusted_trend_20d']=trend;lib['return_sign_balance_20obs']=balance;lib['realized_volatility_20obs']=vol20;lib['relative_volume_participation_20d']=np.log(V/V.rolling(20,min_periods=15).mean())
mx=-1;who='';cells=0
for n,o in lib.items():
 q=pd.concat([f.stack(),o.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if not len(q):print('LIBRARY',n,'MISSING_EVIDENCE');continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman');print(f'LIBRARY {n} rho={rho:.6f} cells={len(q)}')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print(f'max_abs_library_correlation={mx:.6f}; closest={who}; closest_cells={cells}; signals_tested={len(lib)}')
f.to_pickle('scripts/miner_1_20301114_continuous_residualized_volnorm_trough_repair_20obs_candidate_signal.pkl')
