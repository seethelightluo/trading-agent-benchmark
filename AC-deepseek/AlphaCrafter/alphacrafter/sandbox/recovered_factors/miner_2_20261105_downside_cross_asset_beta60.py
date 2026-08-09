"""miner_2 validation: downside cross-asset beta (60d), endpoint 2026-11-04."""
import glob,json
import numpy as np
import pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-11-04')
p=pd.DataFrame({a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float) for a in A});r=p.pct_change()
vol=pd.DataFrame({a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END].get('volume',pd.Series(np.nan)).astype(float) for a in A})
# Single idea: negative 60-day beta to equal-weight cross-asset market, estimated only on market-down sessions.
# High ranks are assets with low downside co-movement (stress insulation).
mkt=r.mean(axis=1,skipna=True); down=mkt.where(mkt<0); f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
 f[a]=-r[a].rolling(60,min_periods=40).cov(down)/down.rolling(60,min_periods=40).var()
short=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
lib={'miner_3_risk_adjusted_trend_20d':short,'miner_1_ravmom_20obs':short,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_2_realized_volatility_20obs':-r.rolling(20,min_periods=15).std(),'miner_1_vol_of_vol_cv20':-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':vol/vol.rolling(20,min_periods=15).mean()}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change();beta=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A}); own=r.rolling(20,min_periods=15).std();vx=pd.DataFrame(index=p.index,columns=A,dtype=float)
for dt in p.index:
 z=pd.DataFrame({'y':beta.loc[dt],'x':own.loc[dt]}).dropna()
 if len(z)>=8 and z.x.var()>0:
  X=np.c_[np.ones(len(z)),z.x];vx.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib['miner_1_residualized_vix_stress_resilience_beta20']=vx
print('FACTOR downside_cross_asset_beta_60d: -(Cov60(asset return, equal-weight market return | market<0)/Var60(market return | market<0))')
print('VALIDATION_END',END.date(),'UNIVERSE',len(A)); metrics={}
for h in [1,5,10,20]:
 vals=[];ns=[];fw=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.x.nunique()>1:vals.append((dt,z.x.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1);metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
 if h==10:
  for n,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_26',x.index>='2025')]:
   q=x[mask];print('REGIME',n,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
rk=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURNOVER',round(np.mean(ts),6),'TURNOVER_DATES',len(ts))
mx=0
for n,s in lib.items():
 z=pd.concat([f.stack().rename('new'),s.stack().rename('old')],axis=1).dropna();rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6));print('DECAY',json.dumps({str(h):v for h,v in metrics.items()}))
