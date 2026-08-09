"""miner_1 one idea: downside-event relative outperformance consistency, trailing 60 observations."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; cutoff=pd.Timestamp('2027-08-11'); C={};Vol={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date);d=d.query('date<=@cutoff').sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce').replace(0,np.nan);Vol[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C);R=P.pct_change(); m=R.median(axis=1); ex=R.sub(m,axis=0)
# State at s-1 identifies broad downside conditions without using return at s.
state=(m < m.rolling(60,min_periods=40).quantile(.35)).shift(1)
# Fraction of the prior 60 eligible stress sessions in which each asset beat median peer return.
def hit(s):
 return ex.where(state).gt(0).rolling(60,min_periods=12).mean().where(state.astype(float).rolling(60,min_periods=12).sum()>=12)
f=hit(state)
fw={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
def ev(x,y):
 z=[];ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   a=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(a):z.append(a);ns.append(len(q))
 z=np.array(z);return {'dates':len(z),'ic':round(float(z.mean()),5) if len(z) else None,'icir':round(float(z.mean()/z.std(ddof=1)),5) if len(z)>1 else None,'hit':round(float((z>0).mean()),4) if len(z) else None,'mean_n':round(float(np.mean(ns)),2) if ns else None,'min_n':min(ns) if ns else None}
print('FACTOR downside_event_relative_outperformance_consistency_60','cutoff',cutoff.date(),'range',P.index.min().date(),P.index.max().date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),5),'state_days',int(state.sum()))
for h,y in fw.items():print('H',h,ev(f,y))
for n,s in [('2020_21',('2020-01-01','2021-12-31')),('2022_23',('2022-01-01','2023-12-31')),('2024_25',('2024-01-01','2025-12-31')),('2026_27',('2026-01-01','2027-08-11'))]:print('REGIME10',n,ev(f.loc[s[0]:s[1]],fw[10].loc[s[0]:s[1]]))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),5))
# Full admitted-library signal reconstruction, contemporaneous pooled Spearman evidence.
def beta(ri,ma,mask=None,w=40):
 z=pd.concat([ri.rename('r'),ma.rename('m')],axis=1)
 if mask=='down':z=z.where(z.m<0)
 if mask=='up':z=z.where(z.m>=0)
 return z.r.rolling(w,min_periods=12).cov(z.m)/z.m.rolling(w,min_periods=12).var()
def othercorr(a,w=40):return R[a].rolling(w,min_periods=25).corr(R.drop(columns=a).median(axis=1))
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill();vr=vix.pct_change(); vs=pd.Series(np.where(vix/vix.shift(20)-1>0,-1.,1.),index=P.index)
trend=P.pct_change(20)/R.rolling(20,min_periods=15).std(); vol20=R.rolling(20,min_periods=15).std();vol40=R.rolling(40,min_periods=30).std()
idio=R.sub(m,axis=0)
# actual or directly equivalent definitions of all effective factors
L={'gradual_vol_contraction_gated_trend':trend*np.tanh(np.clip(-np.log(vol20/vol40),-2,2)),
'down_up_beta_asym60':pd.DataFrame({a:beta(R[a],m,'down',60)-beta(R[a],m,'up',60) for a in A}),
'rel_volume20':np.log(pd.DataFrame(Vol)/pd.DataFrame(Vol).rolling(20,min_periods=15).mean()),
'quiet_path_efficiency':P.pct_change(20).abs()/R.abs().rolling(20,min_periods=15).sum()*(1-vol20.rank(pct=True)),
'inverse_idio_vol20':-idio.rolling(20,min_periods=15).std(),'risk_adjusted_trend20':trend,
'downside_excess_median40':ex.where(state).rolling(40,min_periods=12).median().where(state.astype(float).rolling(40,min_periods=12).sum()>=12),
'low_commonality40':pd.DataFrame({a:-othercorr(a) for a in A}),'ravmom20':trend,
'commonality_expansion40':pd.DataFrame({a:othercorr(a,20).rolling(20,min_periods=15).mean()-othercorr(a,20).shift(20).rolling(20,min_periods=15).mean() for a in A}),
'downside_beta40':pd.DataFrame({a:beta(R[a],m,'down',40) for a in A}),
'inverse_lag1_ac20':-R.rolling(20,min_periods=15).corr(R.shift(1)),
'vol_transition_serial':-R.rolling(20,min_periods=15).corr(R.shift(1))*np.clip(np.log(R.rolling(5,min_periods=4).std()/vol20),-2,2),
'vix_conditioned_trend':trend.mul(vs,axis=0),'stable_liquidity':-np.log(pd.DataFrame(Vol)/pd.DataFrame(Vol).rolling(20,min_periods=15).mean()).rolling(20,min_periods=15).std(),
'vix_up_beta':pd.DataFrame({a:-beta(R[a],vr,'up',40) for a in A}),'volnorm_reversal5':-P.pct_change(5)/R.rolling(5,min_periods=4).std(),'skew60':R.rolling(60,min_periods=40).skew(),'volscaled_reversal1':-R/vol20}
mx=-1;who=''
for n,x in L.items():
 q=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=spearmanr(q.f,q.x).statistic if len(q)>2 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),5))
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,5),'MOST',who)
