"""One idea: commonality-neutral stress-rebound participation (60 observations).
Residualises the 60d mean return on broad stress-rebound events against contemporaneous
low-commonality and downside-beta-resilience ranks cross-sectionally, isolating recovery
participation beyond defensive/correlation exposure.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cutoff=pd.Timestamp('2027-05-19');H=[1,5,10,20]
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.replace(0,np.nan) for a in A});r=P.pct_change();med=r.median(axis=1)
event=(med.shift(1)<=med.shift(1).rolling(60,min_periods=40).quantile(.20))&(med>0);n=event.astype(float).rolling(60,min_periods=40).sum();base=r.where(event,0).rolling(60,min_periods=40).sum().div(n,axis=0).where(n>=3)
def beta(x,m,cond):
 z=pd.concat([x.rename('x'),m.rename('m')],axis=1).where(lambda q:q.m<0 if cond=='down' else q.m>0)
 return z.x.rolling(40,min_periods=12).cov(z.m)/z.m.rolling(40,min_periods=12).var()
lowcommon=pd.DataFrame({a:-r[a].rolling(40,min_periods=25).corr(r.drop(columns=a).median(axis=1)) for a in A})
downbeta=pd.DataFrame({a:-beta(r[a],med,'down') for a in A})
# Each date: rank inputs and take OLS residual, requiring adequate cross-sectional breadth.
f=pd.DataFrame(np.nan,index=P.index,columns=A)
for t in P.index:
 q=pd.concat([base.loc[t],lowcommon.loc[t],downbeta.loc[t]],axis=1).dropna()
 if len(q)>=8:
  y=q.iloc[:,0].rank().to_numpy(); X=np.c_[np.ones(len(q)),q.iloc[:,1].rank().to_numpy(),q.iloc[:,2].rank().to_numpy()]
  f.loc[t,q.index]=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
def ev(x,y):
 z=[];ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z);return {'ic_dates':len(z),'ic':float(z.mean()) if len(z) else None,'icir':float(z.mean()/z.std(ddof=1)) if len(z)>1 else None,'hit_ratio':float((z>0).mean()) if len(z) else None,'mean_valid_instruments':float(np.mean(ns)) if len(z) else None,'min_valid_instruments':int(min(ns)) if len(z) else None}
fw={h:P.shift(-h)/P-1 for h in H}
print('FACTOR commonality_neutral_stress_rebound_participation_60 cutoff',cutoff.date(),'assets',len(A));print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',float(f.notna().stack().mean()),'event_days',int(event.sum()))
for h in H:print('H',h,ev(f,fw[h]))
for nm,sp in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027YTD',('2027-01-01','2027-05-19'))]:print('REGIME10',nm,ev(f.loc[sp[0]:sp[1]],fw[10].loc[sp[0]:sp[1]]))
print('TURNOVER',float(f.rank(axis=1,pct=True).diff().abs().stack().mean()))
# Complete admitted library reconstructed using definitions; this is mandatory correlation evidence.
V=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').volume.replace(0,np.nan) for a in A});rv=np.log(V/V.rolling(20,min_periods=15).mean());mom=P.pct_change(20)/r.rolling(20,min_periods=15).std();ac=-r.rolling(20,min_periods=15).corr(r.shift(1));vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.pct_change();sg=pd.Series(np.where(vix/vix.shift(20)-1>0,-1.,1.),index=vix.index)
L={'ravmom':mom,'relvol':rv,'rev5':-P.pct_change(5)/r.rolling(5,min_periods=4).std(),'rev1':-r/r.rolling(20,min_periods=15).std(),'quiet':(P.pct_change(20).abs()/r.abs().rolling(20,min_periods=15).sum())*(1-r.rolling(20,min_periods=15).std().rolling(60,min_periods=40).rank(pct=True)),'vixtrend':mom.mul(sg,axis=0),'vixbeta':pd.DataFrame({a:-beta(r[a],vix,'up') for a in A}),'lag1':ac,'idioinvvol':-(r.sub(med,axis=0)).rolling(20,min_periods=15).std(),'downbeta':downbeta,'voltransition':ac*np.log(r.rolling(5,min_periods=4).std()/r.rolling(20,min_periods=15).std()).clip(-2,2),'stableliq':-rv.rolling(20,min_periods=15).std(),'skew60':r.rolling(60,min_periods=40).skew(),'asymbeta':pd.DataFrame({a:beta(r[a],med,'up')-beta(r[a],med,'down') for a in A}),'lowcommon':lowcommon}
mx=0;who=''
for name,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna();rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);print('LIBCORR',name,'cells',len(q),'rho',rho)
 if abs(rho)>mx:mx=abs(rho);who=name
print('MAX_ABS_LIBRARY_CORRELATION',mx,'CLOSEST',who)
