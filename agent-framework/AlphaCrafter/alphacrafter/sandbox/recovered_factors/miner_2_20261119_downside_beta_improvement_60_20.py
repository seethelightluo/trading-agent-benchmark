"""miner_2: validate 60d downside-beta improvement, endpoint 2026-11-18."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-11-18')
def close(a):
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return d.loc[:END,'close'].astype(float)
p=pd.DataFrame({a:close(a) for a in A}); r=p.pct_change(); m=r.mean(axis=1); down=m.where(m<0)
# One factor idea: improvement in downside beta. Positive = downside co-movement declined versus 20 sessions ago.
b=pd.DataFrame({a:r[a].rolling(60,min_periods=20).cov(down)/down.rolling(60,min_periods=20).var() for a in A})
f=b.shift(20)-b
# admitted signal reconstructions
vol=pd.DataFrame({a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END].get('volume',pd.Series(dtype=float)) for a in A})
rv=r.rolling(20,min_periods=15).std(); short=(p/p.shift(20)-1)/rv
lib={'miner_3_risk_adjusted_trend_20d':short,'miner_1_ravmom_20obs':short,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_2_realized_volatility_20obs':-rv,'miner_1_vol_of_vol_cv20':-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':vol/vol.rolling(20,min_periods=15).mean()}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change(); vb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A});vx=pd.DataFrame(index=p.index,columns=A)
for dt in p.index:
 z=pd.DataFrame({'y':vb.loc[dt],'x':rv.loc[dt]}).dropna()
 if len(z)>=8 and z.x.var()>0:
  X=np.c_[np.ones(len(z)),z.x];vx.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib['miner_1_residualized_vix_stress_resilience_beta20']=vx
print('FACTOR downside_beta_improvement_60_20 = beta_downside_60(t-20) - beta_downside_60(t)');print('VALIDATION_END',END.date(),'UNIVERSE',len(A));out={}
for h in [1,5,10,20]:
 vals=[]; ns=[];fw=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.x.nunique()>1: vals.append((dt,z.x.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1);out[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in out[h].items()}))
 if h==10:
  for n,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
   q=x[mask];print('REGIME',n,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'TURNOVER',round(float(np.mean(turn)),6),'TURNOVER_DATES',len(turn))
mx=0
for n,s in lib.items():
 z=pd.concat([f.stack().rename('a'),s.stack().rename('b')],axis=1).dropna();rho=z.a.corr(z.b,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6));print('DECAY',json.dumps(out))
