"""miner_2 -- one candidate: residual upside serial reversal (60d), validated to 2027-12-01."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-01')
def ld(a,c): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,c].astype(float)
p=pd.DataFrame({a:ld(a,'close') for a in A}); v=pd.DataFrame({a:ld(a,'volume') for a in A}); r=p.pct_change(); m=r.mean(axis=1); own=r.rolling(20,min_periods=15).std()
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/y.rolling(w,min_periods=n).var() for a in A})
def resid(y,*xs):
 o=pd.DataFrame(np.nan,index=y.index,columns=A)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(k):x.loc[t] for k,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]: o.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
b60=beta(r,m,60,40); e=r-b60.mul(m,axis=0); pos=e.clip(lower=0)
# Candidate: high score is weak persistence / reversal among positive market-neutral residual outcomes.
f=-pd.DataFrame({a:pos[a].rolling(60,min_periods=45).corr(pos[a].shift(1)) for a in A})
# Reconstruct all 18 currently admitted signals for mandatory pooled signal-correlation evidence.
trend=(p/p.shift(20)-1)/own; lib={}
lib['miner_3_risk_adjusted_trend_20d']=trend;lib['miner_1_ravmom_20obs']=trend
lib['miner_1_volnorm_reversal_5obs']=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
lib['miner_1_vol_of_vol_cv20']=r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean()
lib['miner_3_relative_volume_participation_20d']=np.log(v/v.rolling(20,min_periods=15).mean())
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_1_residualized_vix_stress_resilience_beta20']=resid(-beta(r,vix,20,15),own)
R=r.to_numpy(); M=m.to_numpy(); db20=np.full(R.shape,np.nan);db120=np.full(R.shape,np.nan)
for t in range(len(r)):
 for w,n,out in [(20,8,db20),(120,8,db120)]:
  if t>=w:
   for k in range(15):
    q=M[t-w+1:t+1]<0;x=R[t-w+1:t+1,k][q];y=M[t-w+1:t+1][q]
    if len(x)>=n and np.var(y)>0:out[t,k]=np.cov(x,y,ddof=1)[0,1]/np.var(y,ddof=1)
lib['miner_2_downside_beta_improvement_120_20']=pd.DataFrame(db120-db20,index=p.index,columns=A)
dd=p/p.rolling(60,min_periods=40).max()-1;br=(dd<-.05).mean(axis=1); sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(br.diff()) for a in A})
lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy
mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20);lib['miner_1_market_beta_contraction_60_20']=b60-beta(r,m,20,15)
tail=-r.where(r<0).rolling(20,min_periods=6).mean()/own;lib['miner_1_residualized_downside_tail_containment_20']=resid(tail,trend,own)
raw=(p/p.shift(10)-1)*(-np.minimum(p/p.rolling(60,min_periods=40).max()-1,0));lib['miner_1_residualized_drawdown_recovery_60_10']=resid(raw,trend,own)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_2_dxy_shock_beta_improvement_60_20']=beta(r,dxy,60,30)-beta(r,dxy,20,12)
lib['miner_3_residual_median_minus_mean_60d']=(e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean())/e.rolling(60,min_periods=40).std()
lib['miner_3_residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std()
neg=e.clip(upper=0);lib['miner_2_residual_downside_serial_reversal_60d']=-pd.DataFrame({a:neg[a].rolling(60,min_periods=45).corr(neg[a].shift(1)) for a in A})
disp=r.std(axis=1);lib['miner_3_residual_dispersion_shock_resilience_60d']=-pd.DataFrame({a:(r[a]-m).rolling(60,min_periods=45).corr(disp.diff()) for a in A})
lib['miner_3_realized_volatility_compression_20_60d']=-r.rolling(20,min_periods=15).std()/r.rolling(60,min_periods=40).std()
skew=((r-r.rolling(20,min_periods=15).mean())/r.rolling(20,min_periods=15).std()).pow(3).rolling(20,min_periods=15).mean();lib['miner_1_residualized_realized_return_skewness_20d']=resid(skew,trend)
# breadth recovery capture
B=(dd<-.05).mean(axis=1); shock=B.diff().clip(lower=0);lib['miner_1_breadth_recovery_capture_60d']=pd.DataFrame({a:e[a].rolling(60,min_periods=40).cov(shock)/shock.rolling(60,min_periods=40).var() for a in A})
print('FACTOR residual_upside_serial_reversal_60d validation_end',END.date(),'universe',len(A),'library',len(lib),'panel',p.index.min().date(),p.index.max().date())
out={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; q=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:q.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(q)); sd=x.std(ddof=1);out[h]={'ic':x.mean(),'icir':x.mean()/sd,'std':sd,'se':sd/np.sqrt(len(x)),'hit':(x>0).mean(),'dates':len(x),'n':np.mean(ns)};print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in out[h].items()}))
 if h==20:
  for n,ma in [('2025_26',(x.index<'2027')),('2027',x.index>='2027')]:
   y=x[ma];print('REGIME',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),6),'n',round(np.mean([len(pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()) for t in y.index]),3))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURNOVER',round(np.mean(to),6),'TURNOVER_DATES',len(to),'LATEST_AVAILABLE',int(f.iloc[-1].notna().sum()))
mx=-1
for n,s in lib.items():
 z=pd.concat([f.stack(),s.stack()],axis=1).dropna();rho=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');print('LIB',n,round(rho,6),len(z))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),who,cells)
PY
python scripts/miner_2_20271202_residual_upside_serial_reversal_60d.py