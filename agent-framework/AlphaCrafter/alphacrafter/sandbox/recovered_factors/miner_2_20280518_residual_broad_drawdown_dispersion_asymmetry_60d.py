"""miner_2: standalone residual broad-drawdown dispersion asymmetry, 60d."""
import json,numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-05-17')
def load(a,col): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,col].astype(float)
p=pd.DataFrame({a:load(a,'close') for a in A});vol=pd.DataFrame({a:load(a,'volume') for a in A});r=p.pct_change();m=r.mean(axis=1)
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/y.rolling(w,min_periods=n).var() for a in A})
b60=beta(r,m,60,40);e=r-b60.mul(m,axis=0);own=r.rolling(20,min_periods=15).std()
# Conditional residual-dispersion asymmetry: stability in broad drawdowns versus advances.
breadth=(r>0).mean(axis=1);stress=breadth<=.40;advance=breadth>=.50
sd_down=e.where(stress).rolling(60,min_periods=20).std();sd_up=e.where(advance).rolling(60,min_periods=20).std()
f=-(sd_down-sd_up)/(e.rolling(60,min_periods=40).std()+1e-12)
# Reconstruct all available admitted signal definitions for correlation evidence.
def residual(y,*xs):
 out=pd.DataFrame(np.nan,index=y.index,columns=y.columns)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(i):x.loc[t] for i,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]:out.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
trend=(p/p.shift(20)-1)/own;lv=np.log(vol.replace(0,np.nan));vs=lv-lv.rolling(20,min_periods=15).mean()
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(vol/vol.rolling(20,min_periods=15).mean())}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_1_residualized_vix_stress_resilience_beta20']=residual(-beta(r,vix,20,15),own)
R=r.to_numpy();M=m.to_numpy();db20=np.full(R.shape,np.nan);db120=np.full(R.shape,np.nan)
for t in range(len(r)):
 for w,n,out in [(20,8,db20),(120,8,db120)]:
  if t>=w:
   for k in range(15):
    q=M[t-w+1:t+1]<0;x=R[t-w+1:t+1,k][q];y=M[t-w+1:t+1][q]
    if len(x)>=n and np.var(y)>0:out[t,k]=np.cov(x,y,ddof=1)[0,1]/np.var(y,ddof=1)
lib['miner_2_downside_beta_improvement_120_20']=pd.DataFrame(db120-db20,index=p.index,columns=A)
dd=p/p.rolling(60,min_periods=40).max()-1;dbreadth=(dd<-.05).mean(axis=1);sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(dbreadth.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy
mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20);lib['miner_1_market_beta_contraction_60_20']=b60-beta(r,m,20,15)
tail=-r.where(r<0).rolling(20,min_periods=6).mean()/own;lib['miner_1_residualized_downside_tail_containment_20']=residual(tail,trend,own);raw=(p/p.shift(10)-1)*(-np.minimum(dd,0));lib['miner_1_residualized_drawdown_recovery_60_10']=residual(raw,trend,own)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_2_dxy_shock_beta_improvement_60_20']=beta(r,dxy,60,30)-beta(r,dxy,20,12)
lib['miner_3_residual_median_minus_mean_60d']=e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean();lib['miner_3_residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std();lib['miner_2_downside_vs_upside_volume_change_60d']=lv.diff().where(r<0).rolling(60,min_periods=12).mean()-lv.diff().where(r>0).rolling(60,min_periods=12).mean()
down=e.clip(upper=0);lib['miner_2_residual_downside_serial_reversal_60d']=pd.DataFrame({a:-down[a].rolling(60,min_periods=45).corr(down[a].shift(1)) for a in A});shock=breadth.diff().clip(lower=0);lib['miner_1_breadth_recovery_capture_60d']=pd.DataFrame({a:e[a].rolling(60,min_periods=40).cov(shock)/shock.rolling(60,min_periods=40).var() for a in A});lib['miner_3_realized_volatility_compression_20_60d']=-(r.rolling(20,min_periods=15).std()/r.rolling(60,min_periods=40).std());lib['miner_1_residualized_realized_return_skewness_20d']=pd.DataFrame({a:e[a].rolling(20,min_periods=15).skew() for a in A});disp=r.std(axis=1,ddof=0).diff();lib['miner_3_residual_dispersion_shock_resilience_60d']=pd.DataFrame({a:-e[a].rolling(60,min_periods=45).corr(disp) for a in A});lib['miner_3_residual_upside_volume_confirmation_60d']=(e.clip(lower=0)*vs.clip(lower=0)).rolling(60,min_periods=18).mean()/e.rolling(60,min_periods=40).std();lib['miner_2_residual_broad_drawdown_resilience_60d']=e.where(stress).rolling(60,min_periods=40).mean()
print('FACTOR residual_broad_drawdown_dispersion_asymmetry_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'admitted_library',len(lib),'stress_frequency',round(float(stress.mean()),6))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:out.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q;print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==10:
  for n,mask in [('2020_22',x.index<'2023'),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_26',(x.index>='2025')&(x.index<'2027')),('2027_28',x.index>='2027')]:
   y=x[mask];print('REGIME',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.nanmean(tos),6),'TURNOVER_DATES',len(tos))
mx=-1;win=None
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);win=n
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',win,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
