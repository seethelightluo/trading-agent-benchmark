"""Miner 1: relative volatility-persistence deviation, one interpretable candidate."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];H=[1,5,10,20];cutoff=pd.Timestamp('2027-08-25');C={};V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date');C[a]=d.close.replace(0,np.nan);V[a]=d.volume.replace(0,np.nan)
r=pd.DataFrame(C).pct_change();m=r.median(axis=1)
# candidate: an asset's short/medium volatility ratio, demeaned by the contemporaneous cross-asset ratio
vr=r.rolling(10,min_periods=8).std()/r.rolling(60,min_periods=40).std()
f=-vr.sub(vr.median(axis=1),axis=0).clip(-4,4) # volatility cooling relative to peers
fw={h:pd.DataFrame(C).shift(-h)/pd.DataFrame(C)-1 for h in H}
def ev(x,y):
 z=[];ns=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z);return len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)),float((z>0).mean()),float(np.mean(ns)),int(np.min(ns))
print('FACTOR relative_volatility_persistence_deviation_10_60 cutoff',cutoff.date(),'assets',len(A),'range',f.index.min().date(),f.index.max().date());print('CELLS',f.notna().sum().sum(),'/',f.size,'coverage',f.notna().stack().mean())
for h in H:print('H',h,'dates IC ICIR hit meanN minN',ev(f,fw[h]))
for name,sl in [('2020_21',slice('2020','2021-12-31')),('2022_23',slice('2022','2023-12-31')),('2024_25',slice('2024','2025-12-31')),('2026_27',slice('2026','2027-08-25'))]:print('REGIME10',name,ev(f.loc[sl],fw[10].loc[sl]))
print('TURNOVER',f.rank(axis=1,pct=True).diff().abs().stack().mean())
# Reconstruct each admitted-library signal, then assess pooled point-in-time Spearman dependence.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.replace(0,np.nan).pct_change();sg=pd.Series(np.where(vix/vix.shift(20)-1>0,-1.,1.),index=vix.index)
def beta(ri,macro,side=None,w=40):
 z=pd.concat([ri.rename('r'),macro.rename('m')],axis=1)
 if side=='down':z=z.where(z.m<0)
 if side=='up':z=z.where(z.m>=0)
 return z.r.rolling(w,min_periods=12).cov(z.m)/z.m.rolling(w,min_periods=12).var()
trend=pd.DataFrame({a:C[a].pct_change(20)/r[a].rolling(20,min_periods=15).std() for a in A});co=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(r.drop(columns=a).median(axis=1)) for a in A})
L={'ravmom20':trend,'relvol20':pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}),'volnormrev5':pd.DataFrame({a:-C[a].pct_change(5)/r[a].rolling(5,min_periods=4).std() for a in A}),'volscaledrev1':pd.DataFrame({a:-r[a]/r[a].rolling(20,min_periods=15).std() for a in A}),'vixtrend':trend.mul(sg,axis=0),'vixshockbeta':pd.DataFrame({a:-beta(r[a],vix,'up') for a in A}),'lag1':pd.DataFrame({a:-r[a].rolling(20,min_periods=15).corr(r[a].shift(1)) for a in A}),'idioinvvol':pd.DataFrame({a:-(r[a]-m).rolling(20,min_periods=15).std() for a in A}),'downside_beta':pd.DataFrame({a:beta(r[a],m,'down') for a in A}),'voltransition':pd.DataFrame({a:-r[a].rolling(20,min_periods=15).corr(r[a].shift(1))*np.log(r[a].rolling(5,min_periods=4).std()/r[a].rolling(20,min_periods=15).std()).clip(-2,2) for a in A}),'stableliq':pd.DataFrame({a:-np.log(V[a]/V[a].rolling(20,min_periods=15).mean()).rolling(20,min_periods=15).std() for a in A}),'skew60':r.rolling(60,min_periods=40).skew(),'betaasym60':pd.DataFrame({a:beta(r[a],m,'down',60)-beta(r[a],m,'up',60) for a in A}),'down_event_mag':r.sub(m,axis=0).where((m<m.rolling(60,min_periods=40).quantile(.35)).shift(1),axis=0).rolling(40,min_periods=12).median(),'commonexp40':co.rolling(20,min_periods=15).mean()-co.shift(20).rolling(20,min_periods=15).mean(),'lowcommon40':-pd.DataFrame({a:r[a].rolling(40,min_periods=25).corr(r.drop(columns=a).median(axis=1)) for a in A})}
mx=0;who=''
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna();rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);print('LIBCORR',n,len(q),rho)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',who)
