"""One idea: cross-asset stress-rebound participation (60 observations).
Score is mean own return on broad-market rebound days immediately following a rolling-tail
cross-asset selloff. It measures realised ability to participate in recoveries, rather than
unconditional momentum, volatility, or return-distribution shape.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2027-05-19'); H=[1,5,10,20]
C={};V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date');C[a]=d.close.replace(0,np.nan);V[a]=d.volume.replace(0,np.nan)
P=pd.DataFrame(C);r=P.pct_change();med=r.median(axis=1)
# Trigger is known at the preceding close; retain windows with at least three realised rebound events.
event=(med.shift(1)<=med.shift(1).rolling(60,min_periods=40).quantile(.20)) & (med>0)
event_count=event.astype(float).rolling(60,min_periods=40).sum()
f=r.where(event,0).rolling(60,min_periods=40).sum().div(event_count,axis=0).where(event_count>=3)
fw={h:P.shift(-h)/P-1 for h in H}
def ev(x,y):
 z=[];ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z)
 return {'ic_dates':len(z),'ic':None if not len(z) else float(z.mean()),'icir':None if len(z)<2 else float(z.mean()/z.std(ddof=1)),'hit_ratio':None if not len(z) else float((z>0).mean()),'mean_valid_instruments':None if not len(z) else float(np.mean(ns)),'min_valid_instruments':None if not len(z) else int(min(ns))}
print('FACTOR stress_rebound_participation_60 cutoff',cutoff.date(),'range',f.index.min().date(),f.index.max().date(),'assets',len(A)); print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',float(f.notna().stack().mean()),'trigger_days',int(event.sum()))
for h in H: print('H',h,ev(f,fw[h]))
for nm,sp in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027YTD',('2027-01-01','2027-05-19'))]: print('REGIME10',nm,ev(f.loc[sp[0]:sp[1]],fw[10].loc[sp[0]:sp[1]]))
print('TURNOVER',float(f.rank(axis=1,pct=True).diff().abs().stack().mean()))
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.replace(0,np.nan).pct_change();sg=pd.Series(np.where(vix/vix.shift(20)-1>0,-1.,1.),index=vix.index); med=r.median(axis=1)
def beta(x,m,cond=None):
 z=pd.concat([x.rename('x'),m.rename('m')],axis=1)
 if cond=='down':z=z.where(z.m<0)
 if cond=='up':z=z.where(z.m>0)
 return z.x.rolling(40,min_periods=12).cov(z.m)/z.m.rolling(40,min_periods=12).var()
def oth(a):return -r[a].rolling(40,min_periods=25).corr(r.drop(columns=a).median(axis=1))
rv=pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A});mom=pd.DataFrame({a:P[a].pct_change(20)/r[a].rolling(20,min_periods=15).std() for a in A});ac=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).corr(r[a].shift(1)) for a in A})
L={'ravmom':mom,'relvol':rv,'rev5':pd.DataFrame({a:-P[a].pct_change(5)/r[a].rolling(5,min_periods=4).std() for a in A}),'rev1':pd.DataFrame({a:-r[a]/r[a].rolling(20,min_periods=15).std() for a in A}),'quiet':pd.DataFrame({a:(P[a].pct_change(20).abs()/r[a].abs().rolling(20,min_periods=15).sum())*(1-r[a].rolling(20,min_periods=15).std().rolling(60,min_periods=40).rank(pct=True)) for a in A}),'vixtrend':mom.mul(sg,axis=0),'vixbeta':pd.DataFrame({a:-beta(r[a],vix,'up') for a in A}),'lag1':ac,'idioinvvol':pd.DataFrame({a:-(r[a]-med).rolling(20,min_periods=15).std() for a in A}),'downbeta':pd.DataFrame({a:-beta(r[a],med,'down') for a in A}),'voltransition':ac*np.log(r.rolling(5,min_periods=4).std()/r.rolling(20,min_periods=15).std()).clip(-2,2),'stableliq':-rv.rolling(20,min_periods=15).std(),'skew60':r.rolling(60,min_periods=40).skew(),'asymbeta':pd.DataFrame({a:beta(r[a],med,'up')-beta(r[a],med,'down') for a in A}),'lowcommon':pd.DataFrame({a:oth(a) for a in A})}
mx=0;who=''
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna();rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);print('LIBCORR',n,'cells',len(q),'rho',rho)
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'CLOSEST',who)
