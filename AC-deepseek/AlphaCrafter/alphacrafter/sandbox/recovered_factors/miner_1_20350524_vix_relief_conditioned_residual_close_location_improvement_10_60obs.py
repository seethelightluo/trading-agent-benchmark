"""Single candidate: VIX-relief-conditioned residual close-location improvement.
Measures improvement in close location after idiosyncratic loss, weighted by a prior
VIX deceleration. It is orthogonalized cross-sectionally to 20d risk-adjusted trend.
All predictor inputs at t use completed bars; forward returns are evaluation only."""
import pandas as pd,numpy as np
from scipy.stats import spearmanr
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read()
src=src.replace("END=pd.Timestamp('2032-06-23')", "END=pd.Timestamp('2035-05-23')")
# obtain p, OHLCV, returns, residual series and macro helper, without old candidate/audit
exec(src.split('# Repair-rank migration')[0])
vix=macro('VIX')
clv=((p-lo)/(hi-lo).replace(0,np.nan)).clip(0,1)
# VIX return is known on t-1 when deciding on t; continuous positive relief intensity.
relief=(-vix.shift(1)).clip(lower=0).div(vix.abs().rolling(60,min_periods=45).std().shift(1).clip(lower=1e-8)).clip(upper=3)
# Today's close-location improvement after yesterday's idiosyncratic loss, averaged 10 bars.
loss=(-res.shift(1)).clip(lower=0)
improve=clv.sub(clv.shift(1)).mul(loss)
base=improve.mul(relief,axis=0).rolling(10,min_periods=7).mean()
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
f=orth(base,trend)
print('FACTOR vix_relief_conditioned_residual_close_location_improvement_10_60obs')
print('EXPRESSION cs_residual(mean_10(max(-VIX_return[t-1],0)/std_60(abs(VIX_return))*max(-residual_return[t-1],0)*(CLV[t]-CLV[t-1])),risk_adjusted_trend_20)')
print('endpoint',p.index.max().date(),'assets',len(A),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H); z=[];ds=[];ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns)
 print('HORIZON',H,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'dates',len(z),'mean_n',round(np.mean(ns),3),'min_n',min(ns),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
best=max(R,key=lambda H:abs(R[H][0].mean())*abs(R[H][0].mean()/R[H][0].std(ddof=1)))
z,ds,_=R[best];print('SELECTED_HORIZON',best)
for nm,st,en in [('2026_2029','2026-01-01','2029-12-31'),('2030_2032','2030-01-01','2032-12-31'),('2033_current','2033-01-01','2035-05-23'),('latest_6m','2034-11-23','2035-05-23')]:
 x=z[(ds>=st)&(ds<=en)];print('REGIME',nm,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rnk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('QUALITY daily_rank_turnover',round(np.mean(turn),6),'comparisons',len(turn),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),8))
# Reconstruct the current 30 admitted signals from the canonical audit set, then
# explicitly require finite paired correlation evidence for every comparator.
candidate=f
exec(src.split('# Full current-library reconstruction')[0]); f=candidate
audit=src[src.index('# Full current-library reconstruction'):];audit=audit[:audit.index('mx=-1')];exec(audit)
# Two later admitted definitions absent in legacy canonical set.
disp=r.std(axis=1); hi_state=disp.shift(1).gt(disp.rolling(60,min_periods=45).median().shift(1)).astype(float); den=hi_state.rolling(60,min_periods=45).sum()
L['dispersion_conditioned_residual_resilience_60obs']=res.shift(1).rank(axis=1,pct=True).mul(hi_state,axis=0).rolling(60,min_periods=45).sum().div(den.replace(0,np.nan),axis=0).where(den>=12)
L['common_trend_conditioned_residual_downside_close_location_improvement_10_60obs']=orth(clv.sub(clv.shift(1)).mul((-res.shift(1)).clip(lower=0)).rolling(10,min_periods=7).mean(),trend)
mx=-1;who=None;cells=0;missing=[]
for n,s in L.items():
 q=pd.concat([f.stack(),s.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman') if len(q)>=8 else np.nan
 if not np.isfinite(rho):missing.append(n)
 elif abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'closest',who,'evidence_cells',cells,'signals_tested',len(L),'missing',missing,'COMPLETE',len(missing)==0)
f.to_pickle('scripts/miner_1_20350524_vix_relief_conditioned_residual_close_location_improvement_10_60obs_signal.pkl')
