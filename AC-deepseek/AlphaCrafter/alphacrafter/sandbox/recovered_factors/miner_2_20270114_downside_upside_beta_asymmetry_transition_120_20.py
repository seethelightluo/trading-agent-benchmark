"""miner_2: validate downside-versus-upside beta asymmetry transition, 120d/20d."""
import glob,json
import numpy as np
import pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-01-13')
P={};V={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END];P[a]=d.close.astype(float);V[a]=d.volume.astype(float)
p=pd.DataFrame(P);r=p.pct_change();vol=pd.DataFrame(V);mkt=r.mean(axis=1);N,K=p.shape
def cb(x,y,pos):
 q=np.isfinite(x)&np.isfinite(y)&((y<0) if pos=='down' else (y>0))
 return np.cov(x[q],y[q],ddof=1)[0,1]/np.var(y[q],ddof=1) if q.sum()>=8 and np.var(y[q],ddof=1)>0 else np.nan
def bet(n,kind):
 o=np.full((N,K),np.nan);R=r.values;M=mkt.values
 for t in range(n-1,N):
  for j in range(K):o[t,j]=cb(R[t-n+1:t+1,j],M[t-n+1:t+1],kind)
 return pd.DataFrame(o,index=p.index,columns=A)
up20,up120,dn20,dn120=bet(20,'up'),bet(120,'up'),bet(20,'down'),bet(120,'down')
f=(dn120-dn20)-(up120-up20)
# all current admitted signals reconstructed
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std(); rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); vov=-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean()
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':rev,'miner_1_vol_of_vol_cv20':vov,'miner_3_relative_volume_participation_20d':np.log(vol/vol.rolling(20,min_periods=15).mean()),'miner_2_downside_beta_improvement_120_20':dn120-dn20}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END].close.astype(float).pct_change();b=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A});own=r.rolling(20,min_periods=15).std();vx=pd.DataFrame(np.nan,index=p.index,columns=A)
for t in p.index:
 z=pd.DataFrame({'y':b.loc[t],'x':own.loc[t]}).dropna()
 if len(z)>=8 and z.x.var()>0:vx.loc[t,z.index]=z.y-np.c_[np.ones(len(z)),z.x]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.x],z.y,rcond=None)[0]
lib['miner_1_residualized_vix_stress_resilience_beta20']=vx
dd=p/p.rolling(60,min_periods=40).max()-1;breadth=(dd<-.05).mean(axis=1);ds=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=ds.shift(20)-ds
ms=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(mkt) for a in A});lib['market_synchronization_increase_60_20']=ms-ms.shift(20)
# residualized drawdown recovery proxy: 10d recovery after 60d drawdown, residualized versus contemporaneous drawdown
rec=p/p.shift(10)-1; d60=p/p.rolling(60,min_periods=45).max()-1; rd=pd.DataFrame(np.nan,index=p.index,columns=A)
for t in p.index:
 z=pd.DataFrame({'y':rec.loc[t],'x':d60.loc[t]}).dropna()
 if len(z)>=8:rd.loc[t,z.index]=z.y-np.c_[np.ones(len(z)),z.x]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.x],z.y,rcond=None)[0]
lib['miner_1_residualized_drawdown_recovery_60_10']=rd
print('FACTOR downside_upside_beta_asymmetry_transition_120_20; END',END.date(),'UNIVERSE',K,'PANEL',p.index.min().date(),p.index.max().date())
MET={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'x':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.x.nunique()>1:out.append((t,z.x.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));sd=x.std(ddof=1);MET[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};print('HORIZON',h,json.dumps({a:round(float(q),6) for a,q in MET[h].items()}))
 for nm,mask in [('2020',x.index<'2021-01-01'),('2021_2022',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_2024',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_2026',(x.index>='2025-01-01')&(x.index<'2027-01-01')),('2027',x.index>='2027-01-01')]:
  q=x[mask]
  if len(q)>1:print('REGIME',h,nm,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
rank=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rank)):
 z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURNOVER',round(float(np.mean(to)),6),'TURNOVER_DATES',len(to))
mx=0
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'LIBRARY_SIZE',len(glob.glob('factors/*.json')));print('DECAY',json.dumps({str(h):v for h,v in MET.items()}))
