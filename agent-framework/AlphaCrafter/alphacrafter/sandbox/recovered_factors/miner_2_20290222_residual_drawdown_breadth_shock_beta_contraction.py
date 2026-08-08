"""Validate one candidate: residual drawdown-breadth shock beta contraction (60d minus 20d)."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-02-21')
def load(a,c): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,c].astype(float)
p=pd.DataFrame({a:load(a,'close') for a in A}); vol=pd.DataFrame({a:load(a,'volume') for a in A}); r=p.pct_change(); m=r.mean(axis=1); own=r.rolling(20,min_periods=15).std()
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/(y.rolling(w,min_periods=n).var()+1e-12) for a in A})
def residual(y,*xs):
 out=pd.DataFrame(np.nan,index=y.index,columns=A)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(i):x.loc[t] for i,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]: out.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
b60=beta(r,m,60,40); e=r-b60.mul(m,axis=0); lv=np.log(vol.replace(0,np.nan)); vs=lv-lv.rolling(20,min_periods=15).mean()
trend=(p/p.shift(20)-1)/own
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(vol/vol.rolling(20,min_periods=15).mean())}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_1_residualized_vix_stress_resilience_beta20']=residual(-beta(r,vix,20,15),own)
R=r.to_numpy();M=m.to_numpy(); db20=np.full(R.shape,np.nan);db120=np.full(R.shape,np.nan)
for t in range(len(r)):
 for w,n,out in [(20,8,db20),(120,8,db120)]:
  if t>=w:
   for k in range(15):
    q=M[t-w+1:t+1]<0;x=R[t-w+1:t+1,k][q];y=M[t-w+1:t+1][q]
    if len(x)>=n and np.var(y)>0:out[t,k]=np.cov(x,y,ddof=1)[0,1]/np.var(y)
lib['miner_2_downside_beta_improvement_120_20']=pd.DataFrame(db120-db20,index=p.index,columns=A)
dd=p/p.rolling(60,min_periods=40).max()-1; breadthdd=(dd<-.05).mean(axis=1); shock=breadthdd.diff()
sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(shock) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy
mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20);lib['miner_1_market_beta_contraction_60_20']=b60-beta(r,m,20,15)
tail=-r.where(r<0).rolling(20,min_periods=6).mean()/own;lib['miner_1_residualized_downside_tail_containment_20']=residual(tail,trend,own)
raw=(p/p.shift(10)-1)*(-np.minimum(p/p.rolling(60,min_periods=40).max()-1,0));lib['miner_1_residualized_drawdown_recovery_60_10']=residual(raw,trend,own)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_2_dxy_shock_beta_improvement_60_20']=beta(r,dxy,60,30)-beta(r,dxy,20,12)
lib['miner_3_residual_median_minus_mean_60d']=e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean();lib['miner_3_residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std();lib['miner_2_downside_vs_upside_volume_change_60d']=lv.diff().where(r<0).rolling(60,min_periods=12).mean()-lv.diff().where(r>0).rolling(60,min_periods=12).mean()
de=e.clip(upper=0);lib['miner_2_residual_downside_serial_reversal_60d']=pd.DataFrame({a:-de[a].rolling(60,min_periods=45).corr(de[a].shift(1)) for a in A});B=(r>0).mean(axis=1); upshock=B.diff().clip(lower=0);lib['miner_1_breadth_recovery_capture_60d']=pd.DataFrame({a:e[a].rolling(60,min_periods=40).cov(upshock)/(upshock.rolling(60,min_periods=40).var()+1e-12) for a in A});lib['miner_3_realized_volatility_compression_20_60d']=-(r.rolling(20,min_periods=15).std()/(r.rolling(60,min_periods=40).std()+1e-12));lib['miner_1_residualized_realized_return_skewness_20d']=e.rolling(20,min_periods=15).skew();disp=r.std(axis=1,ddof=0).diff();lib['miner_3_residual_dispersion_shock_resilience_60d']=pd.DataFrame({a:-e[a].rolling(60,min_periods=45).corr(disp) for a in A})
uv=e.clip(lower=0)*vs.clip(lower=0);dv=(-e).clip(lower=0)*vs;dsv=(-e).clip(lower=0)*vs
lib['miner_3_residual_upside_volume_confirmation_60d']=uv.rolling(60,min_periods=18).mean()/(e.rolling(60,min_periods=40).std()+1e-12);lib['miner_3_residual_upside_volume_confirmation_deceleration_20_60d']=-(uv.rolling(20,min_periods=12).mean()/(e.rolling(20,min_periods=15).std()+1e-12)-uv.rolling(60,min_periods=25).mean()/(e.rolling(60,min_periods=40).std()+1e-12));lib['miner_3_residual_downside_volume_confirmation_deceleration_20_60d']=-(dv.rolling(20,min_periods=12).mean()/(e.rolling(20,min_periods=15).std()+1e-12)-dv.rolling(60,min_periods=25).mean()/(e.rolling(60,min_periods=40).std()+1e-12));lib['miner_3_residual_downside_signed_volume_pressure_deceleration_20_60d']=-(dsv.rolling(20,min_periods=12).mean()/(e.rolling(20,min_periods=15).std()+1e-12)-dsv.rolling(60,min_periods=25).mean()/(e.rolling(60,min_periods=40).std()+1e-12))
breadth=(r>0).mean(axis=1).diff(); b20=pd.DataFrame({a:e[a].rolling(20,min_periods=14).cov(breadth)/(breadth.rolling(20,min_periods=14).var()+1e-12) for a in A});b60=pd.DataFrame({a:e[a].rolling(60,min_periods=42).cov(breadth)/(breadth.rolling(60,min_periods=42).var()+1e-12) for a in A});lib['miner_3_residual_breadth_shock_sensitivity_expansion_20_60d']=b20-b60
# Candidate: residual loading on broad drawdown-breadth shock has contracted from structural 60d level.
f=beta(e,shock,60,40)-beta(e,shock,20,14)
print('FACTOR residual_drawdown_breadth_shock_beta_contraction_60_20 validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q):out.append((t,q));ns.append(len(z))
 x=pd.Series(dict(out));ics[h]=x; sd=x.std(ddof=1); metrics[h]=[x.mean(),x.mean()/sd,(x>0).mean(),len(x),np.mean(ns)];print('HORIZON',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'HIT',round((x>0).mean(),6),'DATES',len(x),'MEAN_N',round(np.mean(ns),3))
for name,mask in [('2025_2026',(ics[20].index>='2025')&(ics[20].index<'2027')),('2027_2029',(ics[20].index>='2027'))]:
 x=ics[20][mask];print('REGIME20',name,'DATES',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'HIT',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.nanmean(tos),6),'TURNOVER_DATES',len(tos))
res=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');res.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(res); print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',n,'rho',round(rho,6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(v[0],6),'icir':round(v[1],6),'dates':v[3]} for h,v in metrics.items()}))
