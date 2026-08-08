"""miner_2: one idea -- market-residual positive-session frequency, 60 days."""
import numpy as np,pandas as pd,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-04-07')
def load(a,col='close',path='../persistent/stock_data/'):
 return pd.read_csv(path+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,col].astype(float)
p=pd.DataFrame({a:load(a) for a in A}); v=pd.DataFrame({a:load(a,'volume') for a in A});r=p.pct_change();m=r.mean(axis=1); vol=r.rolling(20,min_periods=15).std()
def beta(x,y,w,n):return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/y.rolling(w,min_periods=n).var() for a in A})
def resid(y,*xs):
 o=pd.DataFrame(np.nan,index=y.index,columns=A)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(i):x.loc[t] for i,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]:o.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
b60=beta(r,m,60,40); e=r-b60.mul(m,axis=0)
# Higher: more frequent positive idiosyncratic return sessions after market exposure removal.
f=(e>0).where(e.notna()).rolling(60,min_periods=40).mean()
trend=(p/p.shift(20)-1)/vol
lib={'miner_1_ravmom_20obs':trend,'miner_3_risk_adjusted_trend_20d':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(v/v.rolling(20,min_periods=15).mean()),'miner_1_market_beta_contraction_60_20':b60-beta(r,m,20,15)}
vix=load('VIX',path='../persistent/index_data/').pct_change();lib['miner_1_residualized_vix_stress_resilience_beta20']=resid(-beta(r,vix,20,15),vol)
# down beta / transition signals
R=r.to_numpy();M=m.to_numpy();d120=np.full(R.shape,np.nan)
for t in range(len(r)):
 if t>=120:
  q=M[t-119:t+1]<0
  for k in range(15):
   x=R[t-119:t+1,k][q];y=M[t-119:t+1][q]
   if len(x)>=8 and np.var(y)>0:d120[t,k]=np.cov(x,y,ddof=1)[0,1]/np.var(y,ddof=1)
d120=pd.DataFrame(d120,index=p.index,columns=A);lib['miner_2_downside_beta_improvement_120_20']=d120.shift(20)-d120
dd=p/p.rolling(60,min_periods=40).max()-1;br=(dd<-.05).mean(axis=1);sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(br.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy
mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20)
tail=-r.where(r<0).rolling(20,min_periods=6).mean()/vol;lib['miner_1_residualized_downside_tail_containment_20']=resid(tail,trend,vol)
raw=(p/p.shift(10)-1)*(-np.minimum(dd,0));lib['miner_1_residualized_drawdown_recovery_60_10']=resid(raw,trend,vol)
dxy=load('DXY',path='../persistent/index_data/').pct_change();lib['miner_2_dxy_shock_beta_improvement_60_20']=beta(r,dxy,60,30)-beta(r,dxy,20,12)
lib['miner_3_residual_median_minus_mean_60d']=(e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean())/e.rolling(60,min_periods=40).std()
lib['miner_3_residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std()
print('FACTOR residual_positive_frequency_60d','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'admitted_library',len(lib))
met={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;zlist=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:zlist.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(zlist));sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};met[h]=q;print('HORIZON',h,json.dumps({k:round(float(w),6) for k,w in q.items()}))
 if h==5:
  for n,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_27',x.index>='2025')]:
   y=x[mask];print('REGIME',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.mean(to),6),'TURNOVER_DATES',len(to))
mx=-1;winner=''
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);winner=n
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in met.items()}))
