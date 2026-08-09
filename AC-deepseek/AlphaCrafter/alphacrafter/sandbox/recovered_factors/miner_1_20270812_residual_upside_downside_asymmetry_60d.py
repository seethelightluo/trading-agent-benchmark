"""miner_1: one candidate -- residual upside/downside asymmetry over 60 sessions."""
import pathlib, json, numpy as np, pandas as pd
src=pathlib.Path('scripts/miner_3_20270422_residual_upside_market_down_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2027-04-22')", "END=pd.Timestamp('2027-08-11')")
exec(src.split("# Mean positive idiosyncratic return only")[0])
trend=(p/p.shift(20)-1)/own
# Continuous residual-distribution resilience: average positive residual return
# minus average negative residual loss, standardized by residual dispersion.
pos=e.clip(lower=0).rolling(60,min_periods=40).mean(); neg=(-e.clip(upper=0)).rolling(60,min_periods=40).mean()
f=(pos-neg)/e.rolling(60,min_periods=40).std()
# Complete current admitted library reconstruction.
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
dd=p/p.rolling(60,min_periods=40).max()-1;breadth=(dd<-.05).mean(axis=1);sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy
mc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A});lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20);lib['miner_1_market_beta_contraction_60_20']=b60-beta(r,m,20,15)
lib['miner_1_residualized_downside_tail_containment_20']=residual(-r.where(r<0).rolling(20,min_periods=6).mean()/own,trend,own)
raw=(p/p.shift(10)-1)*(-np.minimum(p/p.rolling(60,min_periods=40).max()-1,0));lib['miner_1_residualized_drawdown_recovery_60_10']=residual(raw,trend,own)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change();lib['miner_2_dxy_shock_beta_improvement_60_20']=beta(r,dxy,60,30)-beta(r,dxy,20,12)
lib['miner_3_residual_median_minus_mean_60d']=e.rolling(60,min_periods=40).median()-e.rolling(60,min_periods=40).mean();lib['miner_3_residual_lower_partial_moment_60d']=-e.clip(upper=0).rolling(60,min_periods=40).mean()/e.rolling(60,min_periods=40).std()
rv=np.log(vol/vol.rolling(20,min_periods=15).mean());lib['miner_2_downside_vs_upside_volume_change_60d']=rv.where(r<0).rolling(60,min_periods=12).mean()-rv.where(r>0).rolling(60,min_periods=12).mean()
improve=breadth.diff()<0;lib['miner_1_breadth_recovery_capture_60d']=r.where(improve,np.nan).rolling(60,min_periods=12).mean()/own
loss=(-e.clip(upper=0));lib['miner_2_residual_downside_serial_reversal_60d']=loss.rolling(20,min_periods=12).mean()-loss.shift(20).rolling(20,min_periods=12).mean()
print('FACTOR residual_upside_downside_asymmetry_60d','cutoff',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;vals=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q;print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==5:
  for name,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_27',x.index>='2025')]:
   y=x[mask];print('REGIME',name,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.mean(tos),6),'TURNOVER_DATES',len(tos))
mx=-1;winner=None;wcells=0
for name,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',name,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);winner=name;wcells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'CELLS',wcells,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
