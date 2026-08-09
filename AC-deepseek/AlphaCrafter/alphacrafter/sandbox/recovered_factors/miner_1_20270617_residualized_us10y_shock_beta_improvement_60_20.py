"""miner_1: yield-response transition residualized against existing macro-response factors.
Higher values represent a reduction in 20d versus 60d sensitivity to US 10-year yield changes,
after cross-sectional removal of DXY-shock transition and broad-market beta contraction."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-06-16')
def load(a,col): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,col].astype(float)
p=pd.DataFrame({a:load(a,'close') for a in A}); vol=pd.DataFrame({a:load(a,'volume') for a in A}); r=p.pct_change(); m=r.mean(axis=1); own=r.rolling(20,min_periods=15).std()
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/y.rolling(w,min_periods=n).var() for a in A})
def residual(y,*xs):
 out=pd.DataFrame(np.nan,index=y.index,columns=y.columns)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(i):x.loc[t] for i,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]: out.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
b60=beta(r,m,60,40); e=r-b60.mul(m,axis=0)
trend=(p/p.shift(20)-1)/own
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(vol/vol.rolling(20,min_periods=15).mean())}
breadth=(p/p.rolling(60,min_periods=40).max()-1<-.05).mean(axis=1);lib['miner_1_breadth_recovery_capture_60d']=(p/p.shift(10)-1)*(1-breadth).reindex(p.index).values[:,None]
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_1_residualized_vix_stress_resilience_beta20']=residual(-beta(r,vix,20,15),own)
R=r.to_numpy();M=m.to_numpy();db20=np.full(R.shape,np.nan);db120=np.full(R.shape,np.nan)
for t in range(len(r)):
 for w,n,out in [(20,8,db20),(120,8,db120)]:
  if t>=w:
   for k in range(15):
    q=M[t-w+1:t+1]<0;x=R[t-w+1:t+1,k][q];y=M[t-w+1:t+1][q]
    if len(x)>=n and np.var(y)>0: out[t,k]=np.cov(x,y,ddof=1)[0,1]/np.var(y,ddof=1)
lib['miner_2_downside_beta_improvement_120_20']=pd.DataFrame(db120-db20,index=p.index,columns=A)
dd=p/p.rolling(60,min_periods=40).max()-1; breadth=(dd<-.05).mean(axis=1); sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy
mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A}); market_transition=b60-beta(r,m,20,15);lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20);lib['miner_1_market_beta_contraction_60_20']=market_transition
tail=-r.where(r<0).rolling(20,min_periods=6).mean()/own;lib['miner_1_residualized_downside_tail_containment_20']=residual(tail,trend,own)
raw=(p/p.shift(10)-1)*(-np.minimum(p/p.rolling(60,min_periods=40).max()-1,0));lib['miner_1_residualized_drawdown_recovery_60_10']=residual(raw,trend,own)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();dxy_transition=beta(r,dxy,60,30)-beta(r,dxy,20,12);lib['miner_2_dxy_shock_beta_improvement_60_20']=dxy_transition
lib['miner_3_residual_median_minus_mean_60d']=e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean();lib['miner_3_residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std()
yield_transition=beta(r,r['US10Y'],60,30)-beta(r,r['US10Y'],20,12)
f=residual(yield_transition,dxy_transition,market_transition)
print('FACTOR residualized_us10y_shock_beta_improvement_60_20 validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'admitted_library',len(lib))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals)); sd=x.std(ddof=1); q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q;print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==10:
  for name,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_27',x.index>='2025')]:
   y=x[mask];print('REGIME',name,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.mean(tos),6),'TURNOVER_DATES',len(tos))
mx=-1;win=None
for name,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',name,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);win=name
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',win,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
