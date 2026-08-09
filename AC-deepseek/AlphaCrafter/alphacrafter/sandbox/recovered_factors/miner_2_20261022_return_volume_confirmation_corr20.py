"""miner_2 one-factor study: return-volume confirmation correlation, endpoint 2026-10-21."""
import glob, json
import numpy as np
import pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-10-21'); P={}; V={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d.close.astype(float); V[a]=d.volume.astype(float)
p=pd.DataFrame(P); r=p.pct_change(); v=pd.DataFrame(V)
# One interpretable idea: 20-session correlation of asset return and its log volume change.
# Positive values identify price advances (and declines) consistently confirmed by participation.
f=r.rolling(20,min_periods=15).corr(np.log(v).diff())
# Recreate admitted signals for mandatory library-independence test.
lib={}
lib['miner_3_risk_adjusted_trend_20d']=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
lib['miner_1_ravmom_20obs']=lib['miner_3_risk_adjusted_trend_20d']
lib['miner_1_volnorm_reversal_5obs']=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
lib['miner_3_relative_volume_participation_20d']=v/v.rolling(20,min_periods=15).mean()
lib['miner_1_vol_of_vol_cv20']=-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END].close.astype(float).pct_change()
beta=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A}); own=r.rolling(20,min_periods=15).std(); resid=pd.DataFrame(np.nan,index=p.index,columns=A)
for dt in p.index:
 z=pd.DataFrame({'y':beta.loc[dt],'x':own.loc[dt]}).dropna()
 if len(z)>=8: resid.loc[dt,z.index]=z.y-np.column_stack([np.ones(len(z)),z.x])@np.linalg.lstsq(np.column_stack([np.ones(len(z)),z.x]),z.y,rcond=None)[0]
lib['miner_1_residualized_vix_stress_resilience_beta20']=resid
print('FACTOR return_volume_confirmation_corr20 = rolling_corr(return_1d, log(volume)_change_1d, 20); high means moves are participation-confirmed')
print('validation_end',END.date(),'universe',len(A))
summary={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; rows=[]; ns=[]
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8: rows.append((dt,z.x.corr(z.y,method='spearman'))); ns.append(len(z))
 x=pd.Series(dict(rows)); sd=x.std(ddof=1); summary[h]=x
 print(f'H{h} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} dates={len(x)} mean_n={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
 if h==10:
  for nm,mask in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-26',x.index>='2025-01-01')]:
   q=x[mask]; print(f' REGIME {nm} n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'coverage_cells={f.notna().mean().mean():.4f} turnover={np.mean(ts):.6f} turnover_dates={len(ts)}')
mx=0
for name,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman'); mx=max(mx,abs(rho)); print(f'LIB {name} rho={rho:.6f} cells={len(z)}')
print('MAX_ABS_LIBRARY_CORRELATION',f'{mx:.6f}','admitted_json_count',len(glob.glob('factors/*.json')))
