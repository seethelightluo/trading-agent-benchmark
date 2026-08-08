"""One idea: rolling return skewness (60 observations).
Higher skewness identifies assets whose recent return distributions have more favorable
right-tail asymmetry, distinct from trend level and volatility magnitude.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=[1,5,10,20]; cutoff=pd.Timestamp('2027-04-21')
C={};V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date');C[a]=d.close.replace(0,np.nan);V[a]=d.volume.replace(0,np.nan)
r=pd.DataFrame(C).pct_change(); f=r.rolling(60,min_periods=40).skew()
fw={h:pd.DataFrame(C).shift(-h)/pd.DataFrame(C)-1 for h in H}
def ev(x,y):
 z=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z)
 return dict(ic_dates=len(z),ic=float(z.mean()),icir=float(z.mean()/z.std(ddof=1)),hit_ratio=float((z>0).mean()),mean_valid_instruments=float(np.mean(ns)),min_valid_instruments=int(min(ns)))
print('FACTOR return_skewness_60 cutoff',cutoff.date(),'range',f.index.min().date(),f.index.max().date(),'assets',len(A));print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',float(f.notna().stack().mean()))
for h in H:print('H',h,ev(f,fw[h]))
for nm,sp in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027YTD',('2027-01-01','2027-04-21'))]:print('REGIME10',nm,ev(f.loc[sp[0]:sp[1]],fw[10].loc[sp[0]:sp[1]]))
print('TURNOVER',float(f.rank(axis=1,pct=True).diff().abs().stack().mean()))
# Effective-library signals, aligned and reconstructed from definitions.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.replace(0,np.nan).pct_change();sg=pd.Series(np.where(vix/vix.shift(20)-1>0,-1.,1.),index=vix.index); med=r.median(axis=1)
def beta(x,m,condition=None):
 z=pd.concat([x.rename('x'),m.rename('m')],axis=1)
 if condition=='down':z=z.where(z.m<0)
 if condition=='up':z=z.where(z.m>0)
 return z.x.rolling(40,min_periods=12).cov(z.m)/z.m.rolling(40,min_periods=12).var()
def othcorr(a):
 other=r.drop(columns=a).median(axis=1);return -r[a].rolling(40,min_periods=25).corr(other)
rv=pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A})
L={'relative_volume':rv,'quiet_path':pd.DataFrame({a:(C[a].pct_change(20).abs()/r[a].abs().rolling(20,min_periods=15).sum())*(1-r[a].rolling(20,min_periods=15).std().rolling(60,min_periods=40).rank(pct=True)) for a in A}),'inverse_idio_vol':pd.DataFrame({a:-(r[a]-med).rolling(20,min_periods=15).std() for a in A}),'ravmom':pd.DataFrame({a:C[a].pct_change(20)/r[a].rolling(20,min_periods=15).std() for a in A}),'commonality':pd.DataFrame({a:othcorr(a) for a in A}),'downside_beta':pd.DataFrame({a:beta(r[a],med,'down') for a in A}),'lag1':pd.DataFrame({a:-r[a].rolling(20,min_periods=15).corr(r[a].shift(1)) for a in A}),'vol_transition':pd.DataFrame({a:-r[a].rolling(20,min_periods=15).corr(r[a].shift(1))*np.log(r[a].rolling(5,min_periods=4).std()/r[a].rolling(20,min_periods=15).std()).clip(-2,2) for a in A}),'vixtrend':pd.DataFrame({a:(C[a].pct_change(20)/r[a].rolling(20,min_periods=15).std()).mul(sg,axis=0) for a in A}),'stable_liquidity':-rv.rolling(20,min_periods=15).std(),'vix_beta':pd.DataFrame({a:-beta(r[a],vix,'up') for a in A}),'rev5':pd.DataFrame({a:-C[a].pct_change(5)/r[a].rolling(5,min_periods=4).std() for a in A}),'rev1':pd.DataFrame({a:-r[a]/r[a].rolling(20,min_periods=15).std() for a in A})}
mx=0; who=''
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna(); rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);print('LIBCORR',n,'cells',len(q),'rho',rho)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'CLOSEST',who)
