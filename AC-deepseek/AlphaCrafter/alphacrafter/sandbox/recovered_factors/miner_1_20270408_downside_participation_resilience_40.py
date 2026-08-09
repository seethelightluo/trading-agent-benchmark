"""One idea: downside participation resilience (40 observations).
High scores identify assets that participate least often in broad cross-asset down days,
distinct from downside beta because it measures event frequency rather than amplitude.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=[1,5,10,20]; cutoff=pd.Timestamp('2027-04-07')
C={};V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date');C[a]=d.close.replace(0,np.nan);V[a]=d.volume.replace(0,np.nan)
r=pd.DataFrame(C).pct_change(); market=r.median(axis=1); down=market<0
# conditional proportion of negative own returns among observable market-down sessions; negated so high=resilient
f=pd.DataFrame({a:-(r[a]<0).where(down).rolling(40,min_periods=12).mean() for a in A})
fw={h:pd.DataFrame({a:C[a].shift(-h)/C[a]-1 for a in A}) for h in H}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]];y=fw[h] if span is None else fw[h].loc[span[0]:span[1]];z=[];nn=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   k=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(k):z.append(k);nn.append(len(q))
 z=np.array(z);return dict(dates=len(z),ic=float(z.mean()),icir=float(z.mean()/z.std(ddof=1)),hit=float((z>0).mean()),mean_n=float(np.mean(nn)),min_n=int(np.min(nn)))
print('FACTOR downside_participation_resilience_40 cutoff',cutoff.date(),'range',f.index.min().date(),f.index.max().date(),'assets',len(A));print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',float(f.notna().stack().mean()))
for h in H:print('H',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027YTD',('2027-01-01','2027-04-07'))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',float(f.rank(axis=1,pct=True).diff().abs().stack().mean()))
# reconstruct all admitted signals for correlation gate
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.replace(0,np.nan).pct_change();sg=pd.Series(np.where(vix/vix.shift(20)-1>0,-1.,1.),index=vix.index)
def beta(ri,m,down=False):
 z=pd.concat([ri.rename('r'),m.rename('m')],axis=1);z=z.where(z.m<0) if down else z;return -(z.r.rolling(40,min_periods=12).cov(z.m)/z.m.rolling(40,min_periods=12).var())
def quiet(a):return (C[a].pct_change(20).abs()/r[a].abs().rolling(20,min_periods=15).sum())*(1-r[a].rolling(20,min_periods=15).std().rolling(60,min_periods=40).apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1]))
L={'ravmom':pd.DataFrame({a:C[a].pct_change(20)/r[a].rolling(20,min_periods=15).std() for a in A}),'relvol':pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}),'rev5':pd.DataFrame({a:-C[a].pct_change(5)/r[a].rolling(5,min_periods=4).std() for a in A}),'rev1':pd.DataFrame({a:-r[a]/r[a].rolling(20,min_periods=15).std() for a in A}),'quiet':pd.DataFrame({a:quiet(a) for a in A}),'vixtrend':pd.DataFrame({a:(C[a].pct_change(20)/r[a].rolling(20,min_periods=15).std()).mul(sg,axis=0) for a in A}),'vixbeta':pd.DataFrame({a:beta(r[a],vix) for a in A}),'lag1':pd.DataFrame({a:-r[a].rolling(20,min_periods=15).corr(r[a].shift(1)) for a in A}),'downbeta':pd.DataFrame({a:beta(r[a],market,True) for a in A}),'transition':pd.DataFrame({a:-r[a].rolling(20,min_periods=15).corr(r[a].shift(1))*np.log(r[a].rolling(5,min_periods=4).std()/r[a].rolling(20,min_periods=15).std()).clip(-2,2) for a in A}),'idio':pd.DataFrame({a:-(r[a]-market).rolling(20,min_periods=15).std() for a in A}),'liquidity':pd.DataFrame({a:np.log(V[a].rolling(20,min_periods=15).mean()/V[a].rolling(60,min_periods=40).mean()) for a in A}),'commonality':pd.DataFrame({a:-r[a].rolling(40,min_periods=25).corr(r.drop(columns=a).median(axis=1)) for a in A})}
mx=0
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna();rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);mx=max(mx,abs(rho));print('LIBCORR',n,'cells',len(q),'rho',rho)
print('MAX_ABS_LIBRARY_CORRELATION',mx)
