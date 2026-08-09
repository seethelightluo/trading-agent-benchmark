"""miner_2 one-idea validation: upside beta improvement 120/20, visible through 2026-12-30."""
import glob,json
import numpy as np
import pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-30')
P={};V={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d.close.astype(float); V[a]=d.volume.astype(float) if 'volume' in d else pd.Series(np.nan,index=d.index)
p=pd.DataFrame(P); r=p.pct_change(); vol=pd.DataFrame(V); mkt=r.mean(axis=1)
def beta_cond(x,y,positive):
 q=np.isfinite(x)&np.isfinite(y)&((y>0) if positive else (y<0))
 return np.cov(x[q],y[q],ddof=1)[0,1]/np.var(y[q],ddof=1) if q.sum()>=8 and np.var(y[q],ddof=1)>0 else np.nan
R=r.to_numpy();M=mkt.to_numpy();N,K=len(p),len(A)
def make_beta(n,pos):
 out=np.full((N,K),np.nan)
 for j in range(n-1,N):
  for k in range(K):out[j,k]=beta_cond(R[j-n+1:j+1,k],M[j-n+1:j+1],pos)
 return pd.DataFrame(out,index=p.index,columns=A)
up20,up120=make_beta(20,True),make_beta(120,True); dn20,dn120=make_beta(20,False),make_beta(120,False)
f=up120-up20 # high = reduced recent upside beta, intended defensive upside independence
# admitted signal reconstructions
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(vol/vol.rolling(20,min_periods=15).mean()),'miner_2_downside_beta_improvement_120_20':dn120-dn20}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END].close.astype(float).pct_change(); beta=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A}); own=r.rolling(20,min_periods=15).std();vx=pd.DataFrame(np.nan,index=p.index,columns=A)
for dt in p.index:
 z=pd.DataFrame({'y':beta.loc[dt],'x':own.loc[dt]}).dropna()
 if len(z)>=8 and z.x.var()>0:
  X=np.c_[np.ones(len(z)),z.x];vx.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib['miner_1_residualized_vix_stress_resilience_beta20']=vx
dd=p/p.rolling(60,min_periods=40).max()-1;breadth=(dd<-.05).mean(axis=1); ds=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=ds.shift(20)-ds
ms=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(mkt) for a in A});lib['market_synchronization_increase_60_20']=ms-ms.shift(20)
print('FACTOR upside_beta_improvement_120_20: beta_up120 - beta_up20, beta_up conditional on positive equal-weight market days; high=recent reduction in upside-market sensitivity')
print('validation_end',END.date(),'universe',K,'panel',p.index.min().date(),p.index.max().date())
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[];ns=[]
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.x.nunique()>1: vals.append((dt,z.x.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
 for nm,mask in [('2020',x.index<'2021-01-01'),('2021_2022',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_2024',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_2026',x.index>='2025-01-01')]:
  q=x[mask]
  if len(q): print('REGIME',h,nm,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,'hit',round((q>0).mean(),4))
rank=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rank)):
 z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('SIGNAL_CELL_COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(float(np.mean(turn)),6),'TURNOVER_DATES',len(turn))
mx=0
for n,s in lib.items():
 z=pd.concat([f.stack().rename('new'),s.stack().rename('old')],axis=1).dropna();rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'LIBRARY_FILE_COUNT',len(glob.glob('factors/*.json')))
print('DECAY',json.dumps({str(h):{k:float(v) for k,v in m.items()} for h,m in metrics.items()}))
