"""miner_2: revalidate one admitted idea: DXY shock-beta improvement 60d-to-20d."""
import json,numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-09-22')
def L(a,c='close'):
 return pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,c].astype(float)
p=pd.DataFrame({a:L(a) for a in A});v=pd.DataFrame({a:L(a,'volume') for a in A});r=p.pct_change(fill_method=None);m=r.mean(axis=1); rv=r.rolling(20,min_periods=15).std()
def beta(x,y,w,n):return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/y.rolling(w,min_periods=n).var() for a in A})
def resid(y,*xs):
 o=pd.DataFrame(np.nan,index=p.index,columns=A)
 for t in p.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(i):q.loc[t] for i,q in enumerate(xs)}}).dropna()
  if len(z)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(z)),z.iloc[:,1:]])==len(xs)+1:o.loc[t,z.index]=z.y-np.c_[np.ones(len(z)),z.iloc[:,1:]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1:]],z.y,rcond=None)[0]
 return o
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change(); f=beta(r,dxy,60,30)-beta(r,dxy,20,12)
# Reconstruct all currently admitted non-candidate signals, so correlation evidence is complete.
trend=(p/p.shift(20)-1)/rv; b60=beta(r,m,60,40); e=r-b60.mul(m,axis=0); dd=p/p.rolling(60,min_periods=40).max()-1;breadth=(dd<-.05).mean(axis=1)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change()
lib={'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(v/v.rolling(20,min_periods=15).mean()),'miner_1_residualized_vix_stress_resilience_beta20':resid(-beta(r,vix,20,15),rv),'miner_1_residualized_drawdown_recovery_60_10':resid((p/p.shift(10)-1)*(-dd.clip(upper=0)),trend,rv),'miner_1_residualized_downside_tail_containment_20':resid(-(-r.where(r<0)).rolling(20,min_periods=6).mean()/rv,trend,rv),'miner_1_market_beta_contraction_60_20':b60-beta(r,m,20,15),'miner_1_breadth_recovery_capture_60d':r.where(breadth.diff()<0).rolling(60,min_periods=12).mean()/rv,'miner_3_residual_median_minus_mean_60d':e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean(),'miner_3_residual_lower_partial_moment_60d':-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std()}
sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy
lv=np.log(v/v.rolling(20,min_periods=15).mean());lib['miner_2_downside_vs_upside_volume_change_60d']=lv.where(r<0).rolling(60,min_periods=12).mean()-lv.where(r>0).rolling(60,min_periods=12).mean(); loss=-e.clip(upper=0);lib['miner_2_residual_downside_serial_reversal_60d']=loss.rolling(20,min_periods=12).mean()-loss.shift(20).rolling(20,min_periods=12).mean()
print('FACTOR dxy_shock_beta_improvement_60_20 REVALIDATION_END',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
M={}
for h in [1,5,10,20]:
 vals=[];ns=[];fw=p.shift(-h)/p-1
 for t in f.index:
  z=pd.DataFrame({'x':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.x.nunique()>1:vals.append((t,z.x.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1);q={'ic':x.mean(),'icir':x.mean()/sd,'hit':(x>0).mean(),'dates':len(x),'mean_n':np.mean(ns),'se':sd/np.sqrt(len(x))};M[h]=q;print('HORIZON',h,json.dumps({k:round(float(z),6) for k,z in q.items()}))
 if h==1:
  for n,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_27',x.index>='2025')]:
   y=x[mask];print('REGIME',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'TURNOVER',round(float(np.mean(to)),6),'TURNOVER_DATES',len(to))
cs=[]
for n,s in lib.items():
 z=pd.concat([f.stack().rename('x'),s.stack().rename('y')],axis=1).dropna();q=z.x.corr(z.y,method='spearman');cs.append((abs(q),n,q,len(z)));print('LIBRARY',n,'rho',round(q,6),'cells',len(z))
q=max(cs);print('MAX_ABS_LIBRARY_CORRELATION',round(q[0],6),'FACTOR',q[1],'COMPLETE',len(cs)==len(lib));print('DECAY',json.dumps({str(h):{k:float(z) for k,z in x.items()} for h,x in M.items()}))
