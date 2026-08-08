"""Miner_2 validation: downside relative-volume participation asymmetry, one idea."""
import json,numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-06-30')
def L(a,c): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,c].astype(float)
p=pd.DataFrame({a:L(a,'close') for a in A});v=pd.DataFrame({a:L(a,'volume') for a in A});r=p.pct_change();m=r.mean(1); volrel=v/v.rolling(60,min_periods=40).mean()
# Higher: relative volume is more elevated on negative sessions than positive sessions over 60d.
f=volrel.where(r<0).rolling(60,min_periods=8).mean()-volrel.where(r>0).rolling(60,min_periods=8).mean()
own=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/own
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/y.rolling(w,min_periods=n).var() for a in A})
def resid(y,*xs):
 out=pd.DataFrame(np.nan,index=y.index,columns=A)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(i):x.loc[t] for i,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]:out.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
# Reconstruct all admitted signals for mandatory library screen.
lib={'risk_adjusted_trend_20d':trend,'ravmom_20obs':trend,'volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'relative_volume_participation_20d':np.log(v/v.rolling(20,min_periods=15).mean())}
vchg=np.log(v.replace(0,np.nan)).diff();lib['downside_vs_upside_volume_change_60d']=vchg.where(r<0).rolling(60,min_periods=12).mean()-vchg.where(r>0).rolling(60,min_periods=12).mean()
breadth=(p/p.rolling(60,min_periods=40).max()-1<-.05).mean(1);lib['breadth_recovery_capture_60d']=(p/p.shift(10)-1)*(1-breadth).values[:,None]
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['vix_stress_resilience_beta20']=resid(-beta(r,vix,20,15),own)
R=r.values;M=m.values; d20=np.full(R.shape,np.nan);d120=np.full(R.shape,np.nan)
for t in range(len(r)):
 for w,n,o in [(20,8,d20),(120,8,d120)]:
  if t>=w:
   q=M[t-w+1:t+1]<0
   for k in range(15):
    x=R[t-w+1:t+1,k][q];y=M[t-w+1:t+1][q]
    if len(x)>=n and np.var(y)>0:o[t,k]=np.cov(x,y,ddof=1)[0,1]/np.var(y,ddof=1)
lib['downside_beta_improvement_120_20']=pd.DataFrame(d120-d20,index=p.index,columns=A)
dd=p/p.rolling(60,min_periods=40).max()-1;br=(dd<-.05).mean(1);sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(br.diff()) for a in A});lib['drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy
mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['market_synchronization_increase_60_20']=mc-mc.shift(20);b60=beta(r,m,60,40);lib['market_beta_contraction_60_20']=b60-beta(r,m,20,15)
tail=-r.where(r<0).rolling(20,min_periods=6).mean()/own;lib['downside_tail_containment_20']=resid(tail,trend,own);raw=(p/p.shift(10)-1)*(-np.minimum(dd,0));lib['drawdown_recovery_60_10']=resid(raw,trend,own)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['dxy_shock_beta_improvement_60_20']=beta(r,dxy,60,30)-beta(r,dxy,20,12)
e=r-b60.mul(m,axis=0);lib['residual_median_minus_mean_60d']=e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean();lib['residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std()
print('FACTOR downside_relative_volume_asymmetry_60d','end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'assets',len(A),'library',len(lib))
out={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;x=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:x.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(x));sd=x.std(ddof=1);q={'ic':x.mean(),'icir':x.mean()/sd,'hit':(x>0).mean(),'dates':len(x),'mean_n':np.mean(ns)};out[h]=q;print('H',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==1:
  for n,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_27',x.index>='2025')]:
   z=x[mask];print('REGIME',n,'dates',len(z),'ic',round(z.mean(),6),'icir',round(z.mean()/z.std(ddof=1),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURNOVER',round(np.mean(to),6),'TURNOVER_DATES',len(to))
mx=-1
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIB',n,round(rho,6),len(z))
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'WITH',who)
