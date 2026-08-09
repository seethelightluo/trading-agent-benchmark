"""miner_2: one-factor validation: improvement in drawdown-synchronization, current visible endpoint."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-12-02')
def series(a,col='close'):
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,col].astype(float) if col in d else pd.Series(dtype=float)
p=pd.DataFrame({a:series(a) for a in A}); r=p.pct_change(fill_method=None); rv=r.rolling(20,min_periods=15).std()
# Factor: recent reduction in co-movement with the daily cross-asset drawdown-breadth shock.
# Breadth is fraction of all tradables below 95% of their prior 60-session high; correlate asset returns to its daily change.
breadth=(p/p.rolling(60,min_periods=40).max()<.95).mean(axis=1).astype(float)
shock=breadth.diff()
corr=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(shock) for a in A})
f=corr.shift(20)-corr
# reconstruct all currently admitted signals exactly enough for independent rank-correlation gate
vol=pd.DataFrame({a:series(a,'volume') for a in A})
trend=(p/p.shift(20)-1)/rv
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,
 'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),
 'miner_1_vol_of_vol_cv20':-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),
 'miner_3_relative_volume_participation_20d':vol/vol.rolling(20,min_periods=15).mean()}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change(); vb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A}); vx=pd.DataFrame(index=p.index,columns=A)
for dt in p.index:
 z=pd.DataFrame({'y':vb.loc[dt],'x':rv.loc[dt]}).dropna()
 if len(z)>=8 and z.x.var()>0: vx.loc[dt,z.index]=z.y-np.c_[np.ones(len(z)),z.x]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.x],z.y,rcond=None)[0]
lib['miner_1_residualized_vix_stress_resilience_beta20']=vx
m=r.mean(axis=1); down=m.where(m<0); beta=pd.DataFrame({a:r[a].rolling(120,min_periods=30).cov(down)/down.rolling(120,min_periods=30).var() for a in A})
lib['miner_2_downside_beta_improvement_120_20']=beta.shift(20)-beta
print('FACTOR drawdown_synchronization_improvement_60_20; END',END.date(),'UNIVERSE',len(A))
out={}
for h in [1,5,10,20]:
 vals=[];ns=[];fw=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.x.nunique()>1: vals.append((dt,z.x.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals),dtype=float); sd=x.std(ddof=1); q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};out[str(h)+'d']=q;print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==20:
  for n,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
   y=x[mask];print('REGIME',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'TURNOVER',round(float(np.mean(ts)),6),'TURNOVER_DATES',len(ts))
mx=0
for name,s in lib.items():
 z=pd.concat([f.stack().rename('x'),s.stack().rename('y')],axis=1).dropna(); rho=z.x.corr(z.y,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',name,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6));print('DECAY',json.dumps(out))
