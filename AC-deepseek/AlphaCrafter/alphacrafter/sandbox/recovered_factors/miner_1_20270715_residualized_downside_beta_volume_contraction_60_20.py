"""miner_1 one idea: residualized downside-beta contraction with downside-volume participation."""
import json,numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-14')
def load(a,c): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,c].astype(float)
p=pd.DataFrame({a:load(a,'close') for a in A});v=pd.DataFrame({a:load(a,'volume') for a in A});r=p.pct_change();m=r.mean(1);vol=r.rolling(20,min_periods=15).std()
def beta(x,y,w,n):return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/y.rolling(w,min_periods=n).var() for a in A})
def residual(y,*xs):
 o=pd.DataFrame(np.nan,index=y.index,columns=y.columns)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(i):x.loc[t] for i,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]:o.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
def negbeta(w,n):
 o=np.full(r.shape,np.nan);R=r.to_numpy();M=m.to_numpy()
 for t in range(len(r)):
  q=M[max(0,t-w+1):t+1]<0;xM=M[max(0,t-w+1):t+1][q]
  if q.sum()>=n and np.var(xM)>0:
   for j in range(15):
    x=R[max(0,t-w+1):t+1,j][q];ok=np.isfinite(x)&np.isfinite(xM)
    if ok.sum()>=n:o[t,j]=np.cov(x[ok],xM[ok],ddof=1)[0,1]/np.var(xM[ok],ddof=1)
 return pd.DataFrame(o,index=p.index,columns=A)
b60=beta(r,m,60,40);e=r-b60.mul(m,axis=0);trend=(p/p.shift(20)-1)/vol;dd=p/p.rolling(60,min_periods=40).max()-1
n120,n60,n20=negbeta(120,8),negbeta(60,12),negbeta(20,6)
volasym=np.log(v.where(r<0).rolling(60,min_periods=25).mean()/v.where(r>0).rolling(60,min_periods=25).mean())
# High raw value: beta to down-market moves has contracted over 60-to-20 days while downside sessions show relatively stronger participation. Residualization removes the existing beta-improvement and volume-asymmetry signals.
raw=(n60-n20)*volasym
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(v/v.rolling(20,min_periods=15).mean()),'miner_1_market_beta_contraction_60_20':b60-beta(r,m,20,15),'miner_2_downside_beta_improvement_120_20':n120-n20,'miner_2_downside_vs_upside_volume_change_60d':volasym}
f=residual(raw,lib['miner_2_downside_beta_improvement_120_20'],volasym,trend,vol)
lib['miner_1_residualized_downside_tail_containment_20']=residual(-r.where(r<0).rolling(20,min_periods=6).mean()/vol,trend,vol);lib['miner_1_residualized_drawdown_recovery_60_10']=residual((p/p.shift(10)-1)*(-np.minimum(dd,0)),trend,vol)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_1_residualized_vix_stress_resilience_beta20']=residual(-beta(r,vix,20,15),vol)
db=(dd<-.05).mean(1);sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(db.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy;mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_2_dxy_shock_beta_improvement_60_20']=beta(r,dxy,60,30)-beta(r,dxy,20,12);breadth=(r>0).mean(1);pos=breadth.diff().where(breadth.diff()>0,0);lib['miner_1_breadth_recovery_capture_60d']=pd.DataFrame({a:e[a].rolling(60,min_periods=40).cov(pos)/pos.rolling(60,min_periods=40).var() for a in A});lib['miner_3_residual_median_minus_mean_60d']=e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean();lib['miner_3_residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std()
print('FACTOR residualized_downside_beta_volume_contraction_60_20','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib));met={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;vals=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1);q={'ic':x.mean(),'icir':x.mean()/sd,'std':sd,'se':sd/np.sqrt(len(x)),'hit':(x>0).mean(),'dates':len(x),'n':np.mean(ns)};met[h]=q;print('HORIZON',h,json.dumps({k:round(float(z),6)for k,z in q.items()}))
 if h==5:
  for name,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_27',x.index>='2025')]:
   y=x[mask];print('REGIME',name,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.mean(turn),6),'TURNOVER_DATES',len(turn));mx=-1
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'CELLS',cells,'DECAY',json.dumps({str(h):{'ic':round(q['ic'],6),'icir':round(q['icir'],6),'dates':q['dates']}for h,q in met.items()}))
