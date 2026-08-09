"""miner_1: one pre-signed idea -- residual loss-dominance, 60 sessions.
Higher score is deliberately defined ex ante as greater idiosyncratic loss dominance:
mean negative residual magnitude divided by mean positive residual magnitude.
"""
import numpy as np, pandas as pd, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-04-21')
def load(a,c):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,c].astype(float)
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/y.rolling(w,min_periods=n).var() for a in A})
def residual(y,*xs):
 out=pd.DataFrame(np.nan,index=y.index,columns=A)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(j):x.loc[t] for j,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]: out.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
p=pd.DataFrame({a:load(a,'close') for a in A}); v=pd.DataFrame({a:load(a,'volume') for a in A}); r=p.pct_change(); m=r.mean(axis=1); own=r.rolling(20,min_periods=15).std(); b60=beta(r,m,60,40)
e=r-b60.mul(m,axis=0)
pos=e.clip(lower=0).rolling(60,min_periods=40).sum()/e.gt(0).rolling(60,min_periods=40).sum()
neg=(-e.clip(upper=0)).rolling(60,min_periods=40).sum()/e.lt(0).rolling(60,min_periods=40).sum()
f=neg/pos # candidate: HIGH = loss-dominant residual profile, specified before test
# Reconstruct every non-deprecated JSON factor exactly enough for an all-library signal comparison.
trend=(p/p.shift(20)-1)/own; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); volvol=r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean()
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':rev,'miner_1_vol_of_vol_cv20':volvol,'miner_3_relative_volume_participation_20d':np.log(v/v.rolling(20,min_periods=15).mean())}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change(); lib['miner_1_residualized_vix_stress_resilience_beta20']=residual(-beta(r,vix,20,15),own)
# downside beta improvement
def downbeta(w,n):
 z=np.full(r.shape,np.nan); R=r.to_numpy();M=m.to_numpy()
 for t in range(w-1,len(r)):
  q=M[t-w+1:t+1]<0
  for k in range(15):
   x=R[t-w+1:t+1,k][q]; y=M[t-w+1:t+1][q]
   if len(x)>=n and np.var(y)>0:z[t,k]=np.cov(x,y,ddof=1)[0,1]/np.var(y,ddof=1)
 return pd.DataFrame(z,index=p.index,columns=A)
db120=downbeta(120,8); db20=downbeta(20,8); lib['miner_2_downside_beta_improvement_120_20']=db120.shift(20)-db120
dd=p/p.rolling(60,min_periods=40).max()-1; breadth=(dd<-.05).mean(axis=1); sync=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A}); lib['miner_2_drawdown_synchronization_improvement_60_20']=sync.shift(20)-sync
mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A}); lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20);lib['miner_1_market_beta_contraction_60_20']=b60-beta(r,m,20,15)
tail=-r.where(r<0).rolling(20,min_periods=6).mean()/own;lib['miner_1_residualized_downside_tail_containment_20']=residual(tail,trend,own)
raw=(p/p.shift(10)-1)*(-np.minimum(p/p.rolling(60,min_periods=40).max()-1,0));lib['miner_1_residualized_drawdown_recovery_60_10']=residual(raw,trend,own)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_2_dxy_shock_beta_improvement_60_20']=beta(r,dxy,60,30)-beta(r,dxy,20,12)
# miner_3 residual median-minus-mean and lower partial moment
sd=e.rolling(60,min_periods=40).std(); lib['miner_3_residual_median_minus_mean_60d']=(e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean())/sd;lib['miner_3_residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/sd
print('FACTOR residual_loss_dominance_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
allm={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; rows=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(rows)); sdic=x.std(ddof=1); q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sdic,'ic_std':sdic,'ic_standard_error':sdic/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};allm[h]=q;print('HORIZON',h,json.dumps({k:round(float(z),6) for k,z in q.items()}))
 if h==5:
  for name,mask in [('2020',(x.index<'2021')),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_27',x.index>='2025')]:
   y=x[mask];print('REGIME',name,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.mean(turn),6),'TURNOVER_DATES',len(turn))
mx=-1;winner=''
for name,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',name,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);winner=name
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in allm.items()}))
