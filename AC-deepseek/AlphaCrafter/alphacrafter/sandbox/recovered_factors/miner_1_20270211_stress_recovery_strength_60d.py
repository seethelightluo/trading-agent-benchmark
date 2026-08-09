"""miner_1 one-idea validation: stress-recovery strength through 2027-02-10."""
import glob,json
import numpy as np
import pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-02-10')
P={};V={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d.close.astype(float); V[a]=d.volume.astype(float) if 'volume' in d else pd.Series(np.nan,index=d.index)
p=pd.DataFrame(P); r=p.pct_change(); volume=pd.DataFrame(V); m=r.mean(axis=1); own=r.rolling(20,min_periods=15).std()
# Candidate: own-vol-normalized return strength only on broad recovery sessions after a recent market drawdown.
mp=(1+m.fillna(0)).cumprod(); ddm=mp/mp.rolling(20,min_periods=15).max()-1; recovery=(ddm.shift(1)<-.025)&(m>0)
den=recovery.astype(float).rolling(60,min_periods=1).sum()
f=(r/own).where(recovery,np.nan).rolling(60,min_periods=1).mean().where(den>=6,axis=0)
def residual(y,*xs):
 out=pd.DataFrame(np.nan,index=y.index,columns=y.columns)
 for dt in y.index:
  z=pd.DataFrame({'y':y.loc[dt],**{str(i):x.loc[dt] for i,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]: out.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
trend=(p/p.shift(20)-1)/own
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(volume/volume.rolling(20,min_periods=15).mean())}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END].close.astype(float).pct_change()
beta=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A});lib['miner_1_residualized_vix_stress_resilience_beta20']=residual(beta,own)
def db(x,y):
 q=np.isfinite(x)&np.isfinite(y)&(y<0)
 return np.cov(x[q],y[q],ddof=1)[0,1]/np.var(y[q],ddof=1) if q.sum()>=8 and np.var(y[q])>0 else np.nan
R=r.to_numpy();M=m.to_numpy();N=len(p);b20=np.full((N,15),np.nan);b120=b20.copy()
for t in range(N):
 for w,out in [(20,b20),(120,b120)]:
  if t+1>=w:
   for k in range(15):out[t,k]=db(R[t-w+1:t+1,k],M[t-w+1:t+1])
lib['miner_2_downside_beta_improvement_120_20']=pd.DataFrame(b120-b20,index=p.index,columns=A)
dd=p/p.rolling(60,min_periods=40).max()-1;breadth=(dd<-.05).mean(axis=1);sync=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sync.shift(20)-sync
corr=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['miner_2_market_synchronization_increase_60_20']=corr-corr.shift(20)
tail=-(r.where(r<0).rolling(20,min_periods=6).mean())/own;lib['residualized_downside_tail_containment_20']=residual(tail,trend,own)
raw=(p/p.shift(10)-1)*(-np.minimum(p/p.rolling(60,min_periods=40).max()-1,0));lib['miner_1_residualized_drawdown_recovery_60_10']=residual(raw,trend,own)
print('FACTOR stress_recovery_strength_60d: trailing-60 mean asset return / 20d own volatility on aggregate positive recovery sessions; recovery requires prior 20d equal-weight drawdown below -2.5%; >=6 sessions.')
print('validation_end',END.date(),'universe',len(A),'panel',p.index.min().date(),p.index.max().date(),'recovery_sessions',int(recovery.sum()),'library_admitted',len(lib))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;ics=[];ns=[]
 for dt in f.index:
  z=pd.DataFrame({'f':f.loc[dt],'fw':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:ics.append((dt,z.f.corr(z.fw,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(ics));sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==5:
  for nm,mask in [('2020_2022',x.index<'2023'),('2023_2024',(x.index>='2023')&(x.index<'2025')),('2025_2026',(x.index>='2025')&(x.index<'2027')),('2027',x.index>='2027')]:
   z=x[mask];print('REGIME',nm,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,'hit',round((z>0).mean(),4) if len(z) else None)
ranks=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('SIGNAL_CELL_COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(float(np.mean(to)),6),'TURNOVER_DATES',len(to))
mx=-1
for n,s in lib.items():
 z=pd.concat([f.stack().rename('a'),s.stack().rename('b')],axis=1).dropna();rho=z.a.corr(z.b,method='spearman');print('LIBRARY',n,'rho',round(rho,6),'cells',len(z));mx=max(mx,abs(rho))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'DECAY',json.dumps({str(h):{k:float(v) for k,v in q.items()} for h,q in metrics.items()}))
