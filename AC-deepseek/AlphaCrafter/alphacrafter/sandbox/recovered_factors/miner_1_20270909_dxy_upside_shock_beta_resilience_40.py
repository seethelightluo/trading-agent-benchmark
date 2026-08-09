"""Miner 1: DXY upside-shock beta resilience, one macro-conditioned risk-resilience idea."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=[1,5,10,20]; cutoff=pd.Timestamp('2027-09-08');C={};R={};V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date');C[a]=d.close.replace(0,np.nan);R[a]=C[a].pct_change();V[a]=d.volume.replace(0,np.nan)
r=pd.DataFrame(R);m=r.median(axis=1)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.replace(0,np.nan).pct_change()
def beta(ri,macro,mask=None,w=40):
 z=pd.concat([ri.rename('r'),macro.rename('m')],axis=1)
 if mask=='up': z=z.where(z.m>0)
 if mask=='down': z=z.where(z.m<0)
 return z.r.rolling(w,min_periods=12).cov(z.m)/z.m.rolling(w,min_periods=12).var()
# Higher score means lower sensitivity to dollar-strengthening shocks, estimated only on DXY-up days.
f=pd.DataFrame({a:-beta(R[a],dxy,'up',40) for a in A})
fw={h:pd.DataFrame({a:C[a].shift(-h)/C[a]-1 for a in A}) for h in H}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]];y=fw[h] if span is None else fw[h].loc[span[0]:span[1]]; z=[];ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z);return {'dates':len(z),'ic':float(z.mean()) if len(z) else None,'icir':float(z.mean()/z.std(ddof=1)) if len(z)>1 else None,'hit':float((z>0).mean()) if len(z) else None,'mean_n':float(np.mean(ns)) if ns else None,'min_n':int(np.min(ns)) if ns else None}
print('FACTOR dxy_upside_shock_beta_resilience_40 cutoff',cutoff.date(),'range',f.index.min().date(),f.index.max().date(),'assets',len(A));print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',float(f.notna().stack().mean()))
for h in H: print('H',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_27',('2025-01-01','2027-09-08'))]: print('REGIME10',n,ev(10,s))
print('TURNOVER',float(f.rank(axis=1,pct=True).diff().abs().stack().mean()))
# Reconstruct all active-library signal families for binding pooled cross-sectional Spearman novelty evidence.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.replace(0,np.nan).pct_change();sg=pd.Series(np.where(vix/vix.shift(20)-1>0,-1.,1.),index=vix.index)
def quiet(a):return (C[a].pct_change(20).abs()/R[a].abs().rolling(20,min_periods=15).sum())*(1-R[a].rolling(20,min_periods=15).std().rolling(60,min_periods=40).apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1]))
common=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(r.drop(columns=a).median(axis=1)) for a in A})
thresh=r.rolling(60,min_periods=40).quantile(.20).shift(1)
L={'ravmom20':pd.DataFrame({a:C[a].pct_change(20)/R[a].rolling(20,min_periods=15).std() for a in A}),'relvol20':pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}),'volnormrev5':pd.DataFrame({a:-C[a].pct_change(5)/R[a].rolling(5,min_periods=4).std() for a in A}),'volscaledrev1':pd.DataFrame({a:-R[a]/R[a].rolling(20,min_periods=15).std() for a in A}),'quietpath':pd.DataFrame({a:quiet(a) for a in A}),'vixtrend':pd.DataFrame({a:C[a].pct_change(20)/R[a].rolling(20,min_periods=15).std() for a in A}).mul(sg,axis=0),'vixshockbeta':pd.DataFrame({a:-beta(R[a],vix,'up') for a in A}),'lag1':pd.DataFrame({a:-R[a].rolling(20,min_periods=15).corr(R[a].shift(1)) for a in A}),'idioinvvol':pd.DataFrame({a:-(R[a]-m).rolling(20,min_periods=15).std() for a in A}),'downside_beta':pd.DataFrame({a:beta(R[a],m,'down') for a in A}),'voltransition':pd.DataFrame({a:-R[a].rolling(20,min_periods=15).corr(R[a].shift(1))*np.log(R[a].rolling(5,min_periods=4).std()/R[a].rolling(20,min_periods=15).std()).clip(-2,2)}),'stableliq':pd.DataFrame({a:-np.log(V[a]/V[a].rolling(20,min_periods=15).mean()).rolling(20,min_periods=15).std() for a in A}),'skew60':pd.DataFrame({a:R[a].rolling(60,min_periods=40).skew() for a in A}),'betaasym60':pd.DataFrame({a:beta(R[a],m,'down',60)-beta(R[a],m,'up',60) for a in A}),'down_event_mag':r.sub(m,axis=0).where((m<m.rolling(60,min_periods=40).quantile(.35)).shift(1),axis=0).rolling(40,min_periods=12).median(),'commonexp40':common.rolling(20,min_periods=15).mean()-common.shift(20).rolling(20,min_periods=15).mean(),'lowertail':-r.lt(thresh).where(thresh.notna()).rolling(40,min_periods=25).mean()}
mx=0;who=''
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna();rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);print('LIBCORR',n,'cells',len(q),'rho',rho)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',who)
