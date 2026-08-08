"""miner_1: validate one continuous cross-asset factor: upside/downside capture asymmetry."""
import glob,json
import numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-02-24')
P={}; V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d.close.astype(float);V[a]=d.volume.astype(float)
p=pd.DataFrame(P);r=p.pct_change(); own=r.rolling(20,min_periods=15).std(); m=r.mean(axis=1)
# One idea: 60d upside/downside capture asymmetry, standardized by own 20d risk.
# Higher signal means stronger typical up-days relative to typical down-days.
pos=r.where(r>0).rolling(60,min_periods=25).mean(); neg=(-r.where(r<0)).rolling(60,min_periods=25).mean()
f=(pos/(neg+1e-12))/own
# Implement all admitted signals for required complete correlation evidence.
def residual(y,*xs):
 out=pd.DataFrame(np.nan,index=y.index,columns=y.columns)
 for dt in y.index:
  z=pd.DataFrame({'y':y.loc[dt],**{str(i):x.loc[dt] for i,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]:out.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
trend=(p/p.shift(20)-1)/own; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); vv=r.rolling(5,min_periods=4).std(); volvol=-vv.rolling(20,min_periods=15).std()/vv.rolling(20,min_periods=15).mean()
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':rev,'miner_3_relative_volume_participation_20d':np.log(pd.DataFrame(V)/pd.DataFrame(V).rolling(20,min_periods=15).mean()),'miner_1_vol_of_vol_cv20':volvol}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END].close.pct_change(); beta=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A});lib['miner_1_residualized_vix_stress_resilience_beta20']=residual(beta,own)
def db(x,y):
 q=np.isfinite(x)&np.isfinite(y)&(y<0);return np.cov(x[q],y[q],ddof=1)[0,1]/np.var(y[q],ddof=1) if q.sum()>=8 and np.var(y[q])>0 else np.nan
R=r.to_numpy();M=m.to_numpy();N=len(p);b20=np.full((N,15),np.nan);b120=b20.copy()
for t in range(N):
 for w,o in [(20,b20),(120,b120)]:
  if t>=w:
   for k in range(15):o[t,k]=db(R[t-w:t,k],M[t-w:t])
lib['miner_2_downside_beta_improvement_120_20']=pd.DataFrame(b120-b20,index=p.index,columns=A)
dd=p/p.rolling(60,min_periods=40).max()-1; breadth=(dd<-.05).mean(axis=1); sync=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sync.shift(20)-sync
co=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['miner_2_market_synchronization_increase_60_20']=co-co.shift(20)
tail=-(r.where(r<0).rolling(20,min_periods=6).mean())/own;lib['miner_1_residualized_downside_tail_containment_20']=residual(tail,trend,own);raw=(p/p.shift(10)-1)*(-np.minimum(dd,0));lib['miner_1_residualized_drawdown_recovery_60_10']=residual(raw,trend,own)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END].close.pct_change();bD20=pd.DataFrame({a:r[a].rolling(20,min_periods=12).cov(dxy)/dxy.rolling(20,min_periods=12).var() for a in A});bD60=pd.DataFrame({a:r[a].rolling(60,min_periods=30).cov(dxy)/dxy.rolling(60,min_periods=30).var() for a in A});lib['miner_2_dxy_shock_beta_improvement_60_20']=bD60-bD20
print('FACTOR upside_downside_capture_asymmetry_60: 60d mean positive return / absolute mean negative return, divided by 20d own volatility. END',END.date(),'UNIVERSE',len(A),'PANEL',p.index.min().date(),p.index.max().date(),'LIBRARY',len(lib))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; xs=[];ns=[]
 for dt in f.index:
  z=pd.DataFrame({'f':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:xs.append((dt,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(xs)); sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q;print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==5:
  for n,mask in [('2020_2022',x.index<'2023'),('2023_2024',(x.index>='2023')&(x.index<'2025')),('2025_2026',(x.index>='2025')&(x.index<'2027')),('2027',x.index>='2027')]:
   z=x[mask];print('REGIME',n,'DATES',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'HIT',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURNOVER',round(np.mean(ts),6),'TURNOVER_DATES',len(ts))
mx=-1
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',n,'RHO',round(rho,6),'CELLS',len(z));mx=max(mx,abs(rho))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'DECAY',json.dumps({str(h):{'ic':q['daily_paper_ic'],'icir':q['daily_paper_icir'],'dates':q['ic_dates']} for h,q in metrics.items()}))
