"""miner_1 single idea: relative drawdown recovery velocity after an own 40d drawdown."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; cutoff=pd.Timestamp('2027-07-28'); C={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']);d=d.query('date<=@cutoff').sort_values('date').set_index('date');C[a]=pd.to_numeric(d.close,errors='coerce').replace(0,np.nan);V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C); R=P.pct_change()
# Own-asset recovery speed: recent 5d rebound, normalized by the magnitude of the prior 40d peak-to-current drawdown.
peak=P.rolling(40,min_periods=30).max(); dd=P/peak-1
prior_dd=dd.shift(5)
f=(P.pct_change(5)/(-prior_dd).clip(lower=.002)).where(prior_dd < prior_dd.rolling(60,min_periods=40).quantile(.45))
fw={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=fw[h] if span is None else fw[h].loc[span[0]:span[1]];z=[];ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   k=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(k):z.append(k);ns.append(len(q))
 z=np.array(z); return dict(dates=len(z),ic=round(float(z.mean()),5) if len(z) else None,icir=round(float(z.mean()/z.std(ddof=1)),5) if len(z)>1 else None,hit=round(float((z>0).mean()),4) if len(z) else None,mean_n=round(float(np.mean(ns)),2) if ns else None,min_n=min(ns) if ns else None)
print('FACTOR relative_drawdown_recovery_velocity_5_40 cutoff',cutoff.date(),'range',P.index.min().date(),P.index.max().date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),4),'state_dates',int(f.notna().any(axis=1).sum()))
for h in fw:print('H',h,ev(h))
for n,s in [('2020_21',('2020-01-01','2021-12-31')),('2022_23',('2022-01-01','2023-12-31')),('2024_25',('2024-01-01','2025-12-31')),('2026_27',('2026-01-01','2027-07-28'))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),5))
# Broad signal screen against major admitted families (pooled aligned factor cells).
m=R.median(axis=1)
def beta(ri,ma,mask=None,w=40):
 z=pd.concat([ri.rename('r'),ma.rename('m')],axis=1)
 if mask=='down':z=z.where(z.m<0)
 if mask=='up':z=z.where(z.m>=0)
 return z.r.rolling(w,min_periods=12).cov(z.m)/z.m.rolling(w,min_periods=12).var()
trend=P.pct_change(20)/R.rolling(20,min_periods=15).std(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill().pct_change();sg=pd.Series(np.where(vix/vix.shift(20)-1>0,-1.,1.),index=P.index)
L={'ravmom':trend,'reversal5':-P.pct_change(5)/R.rolling(5,min_periods=4).std(),'reversal1':-R/R.rolling(20,min_periods=15).std(),'lag1':-R.rolling(20,min_periods=15).corr(R.shift(1)),'idioinvvol':-(R.sub(m,axis=0)).rolling(20,min_periods=15).std(),'skew60':R.rolling(60,min_periods=40).skew(),'vixtrend':trend.mul(sg,axis=0),'vixupbeta':pd.DataFrame({a:-beta(R[a],vix,'up') for a in A}),'downbeta':pd.DataFrame({a:beta(R[a],m,'down') for a in A}),'betaasym':pd.DataFrame({a:beta(R[a],m,'down',60)-beta(R[a],m,'up',60) for a in A})}
mx=0;who=''
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna(); rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); print('LIBCORR_PROXY',n,'cells',len(q),'rho',round(rho,5));
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION_PROXY',round(mx,5),'MOST',who)
