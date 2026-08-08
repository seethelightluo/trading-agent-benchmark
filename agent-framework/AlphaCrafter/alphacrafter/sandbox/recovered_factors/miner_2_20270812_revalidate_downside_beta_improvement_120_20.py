"""Revalidate one admitted idea: downside beta improvement (120d minus 20d)."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-08-12')
def px(a,c='close'): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,c].astype(float)
p=pd.DataFrame({a:px(a) for a in A}); v=pd.DataFrame({a:px(a,'volume') for a in A}); r=p.pct_change(); m=r.mean(axis=1); own=r.rolling(20,min_periods=15).std()
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/y.rolling(w,min_periods=n).var() for a in A})
def res(y,*xs):
 o=pd.DataFrame(np.nan,index=y.index,columns=A)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(i):q.loc[t] for i,q in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]:o.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
# candidate
R=r.to_numpy();M=m.to_numpy(); db={}
for w,n in [(20,8),(120,8)]:
 z=np.full(R.shape,np.nan)
 for t in range(w-1,len(r)):
  q=M[t-w+1:t+1]<0
  for k in range(15):
   x=R[t-w+1:t+1,k][q]; y=M[t-w+1:t+1][q]
   if len(x)>=n and np.var(y)>0:z[t,k]=np.cov(x,y,ddof=1)[0,1]/np.var(y,ddof=1)
 db[w]=pd.DataFrame(z,index=p.index,columns=A)
f=db[120].shift(20)-db[20]
# every admitted factor signal
b60=beta(r,m,60,40); e=r-b60.mul(m,axis=0); trend=(p/p.shift(20)-1)/own
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(v/v.rolling(20,min_periods=15).mean()),'miner_1_market_beta_contraction_60_20':b60-beta(r,m,20,15)}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_1_residualized_vix_stress_resilience_beta20']=res(-beta(r,vix,20,15),own)
dd=p/p.rolling(60,min_periods=40).max()-1; breadth=(dd<-.05).mean(axis=1); sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy;mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20)
tail=-r.where(r<0).rolling(20,min_periods=6).mean()/own;lib['miner_1_residualized_downside_tail_containment_20']=res(tail,trend,own);raw=(p/p.shift(10)-1)*(-np.minimum(dd,0));lib['miner_1_residualized_drawdown_recovery_60_10']=res(raw,trend,own)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_2_dxy_shock_beta_improvement_60_20']=beta(r,dxy,60,30)-beta(r,dxy,20,12);lib['miner_3_residual_median_minus_mean_60d']=e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean();lib['miner_3_residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std(); neg=e.clip(upper=0);lib['miner_2_residual_downside_serial_reversal_60d']=-pd.DataFrame({a:neg[a].rolling(60,min_periods=45).corr(neg[a].shift(1)) for a in A})
# volume asymmetry
lv=np.log(v).diff(); lib['downside_vs_upside_volume_change_60d']=pd.DataFrame({a:lv[a].where(r[a]<0).rolling(60,min_periods=12).mean()-lv[a].where(r[a]>0).rolling(60,min_periods=12).mean() for a in A})
# breadth recovery
up=breadth.diff().clip(lower=0);lib['miner_1_breadth_recovery_capture_60d']=pd.DataFrame({a:e[a].rolling(60,min_periods=40).cov(up)/up.rolling(60,min_periods=40).var() for a in A})
print('FACTOR downside_beta_improvement_120_20 END',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
out={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; z=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:z.append((t,q.f.corr(q.y,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(z)); sd=x.std(ddof=1); d={'ic':x.mean(),'icir':x.mean()/sd,'std':sd,'se':sd/np.sqrt(len(x)),'hit':(x>0).mean(),'dates':len(x),'n':np.mean(ns)};out[h]=d;print('H',h,json.dumps({k:round(float(v),6) for k,v in d.items()}))
 if h==20:
  for name,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_27',x.index>='2025')]:
   y=x[mask];print('REGIME',name,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURNOVER',round(np.mean(to),6),'TURNOVER_DATES',len(to))
mx=-1
for n,s in lib.items():
 q=pd.concat([f.stack(),s.stack()],axis=1).dropna();rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman');print('LIB',n,round(rho,6),len(q))
 if abs(rho)>mx:mx=abs(rho);win=n
print('MAX',round(mx,6),win,'DECAY',json.dumps({str(h):{'ic':round(d['ic'],6),'icir':round(d['icir'],6),'dates':d['dates']} for h,d in out.items()}))
