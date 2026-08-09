"""miner_1 one-idea validation: 20d upside/downside payoff asymmetry, through prior close 2026-10-21."""
import glob
import numpy as np
import pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-10-21'); P={}; V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d['close'].astype(float); V[a]=d['volume'].astype(float)
p=pd.DataFrame(P); r=p.pct_change(); vol=pd.DataFrame(V)
# One idea: upside/downside payoff asymmetry. Mean gain on up days divided by absolute mean loss on down days in 20 days.
# High scores distinguish assets whose advances have outweighed their setbacks, rather than merely high total return.
up=r.where(r>0).rolling(20,min_periods=12).mean()
down=(-r.where(r<0)).rolling(20,min_periods=12).mean()
f=np.log((up+1e-5)/(down+1e-5))
# Reconstruct live library signals for mandatory full-panel Spearman independence test.
lib={}
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
lib['miner_3_risk_adjusted_trend_20d']=trend; lib['miner_1_ravmom_20obs']=trend
lib['miner_1_volnorm_reversal_5obs']=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
lib['miner_2_realized_volatility_20obs']=-r.rolling(20,min_periods=15).std()
lib['miner_1_vol_of_vol_cv20']=-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean()
lib['miner_3_relative_volume_participation_20d']=vol/vol.rolling(20,min_periods=15).mean()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]['close'].astype(float).pct_change()
beta=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A})
own=r.rolling(20,min_periods=15).std(); resid=pd.DataFrame(np.nan,index=p.index,columns=A)
for dt in p.index:
 z=pd.DataFrame({'y':beta.loc[dt],'x':own.loc[dt]}).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.x]; resid.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib['miner_1_residualized_vix_stress_resilience_beta20']=resid
print('FACTOR upside_downside_payoff_asymmetry_20: log(mean positive daily return / abs(mean negative daily return)) over 20 observations; higher=better payoff asymmetry')
print('VALIDATION_END',END.date(),'universe',len(A),'factor_dates',f.index.min().date(),f.index.max().date())
allmetrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; obs=[]; ns=[]
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8: obs.append((dt,z.x.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(obs)); sd=x.std(ddof=1); allmetrics[h]=x
 print(f'H{h} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} dates={len(x)} mean_n={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
# Regimes at strongest long horizon, avoids selecting a date-specific result.
x=allmetrics[20]
for name,mask in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
 q=x[mask]; print(f'REGIME20 {name} n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rf=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rf)):
 z=pd.concat([rf.iloc[i-1],rf.iloc[i]],axis=1).dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'coverage_cells={f.notna().mean().mean():.4f} mean_valid_names={f.notna().sum(axis=1).mean():.2f} turnover_dates={len(turns)} rank_turnover={np.mean(turns):.6f}')
mx=0
for name,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman'); mx=max(mx,abs(rho));print(f'LIB {name} rho={rho:.6f} cells={len(z)}')
print('library_json_files',len(glob.glob('factors/*.json')),'MAX_ABS_LIBRARY_CORRELATION',f'{mx:.6f}')
