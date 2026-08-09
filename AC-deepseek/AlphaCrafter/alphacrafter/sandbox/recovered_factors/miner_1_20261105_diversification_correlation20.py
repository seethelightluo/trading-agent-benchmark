"""miner_1 20261105: cross-asset diversification score, 20 own observations.
Higher score = lower trailing correlation to an equal-weight cross-asset benchmark.
"""
import json, glob
import numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-11-04'
p={}; v={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 p[a]=d.close.astype(float);v[a]=d.volume.astype(float)
r=pd.DataFrame({a:p[a].pct_change() for a in A}).sort_index()
# Each date: average of pairwise correlations of asset's own last 20 observed returns with all peers' returns on matching dates.
def diversifier(a):
 out=[]
 x=p[a].pct_change()
 for t in r.index:
  xx=x.loc[:t].dropna().tail(20)
  if len(xx)<15: out.append(np.nan);continue
  cs=[]
  for b in A:
   if b==a:continue
   z=pd.concat([xx.rename('x'),p[b].pct_change().rename('y')],axis=1).dropna()
   if len(z)>=15 and z.x.std()>0 and z.y.std()>0:cs.append(z.x.corr(z.y))
  out.append(-np.mean(cs) if len(cs)>=8 else np.nan)
 return pd.Series(out,index=r.index)
f=pd.DataFrame({a:diversifier(a) for a in A})
# Recreate active library signals under same timing.
rv=pd.DataFrame({a:p[a].pct_change().rolling(20,min_periods=15).std() for a in A}).reindex(r.index)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change()
def vb(x):
 z=pd.concat([x.rename('x'),vix.rename('v')],axis=1).dropna();return z.x.rolling(20,min_periods=15).cov(z.v)/z.v.rolling(20,min_periods=15).var()
raw=pd.DataFrame({a:-vb(r[a]) for a in A}).reindex(r.index)
def resid(y,x):
 z=pd.concat([y.rename('y'),x.rename('x')],axis=1).dropna();o=pd.Series(np.nan,index=A)
 if len(z)>=8 and z.x.std()>0:
  X=np.c_[np.ones(len(z)),z.x];o[z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
vixres=pd.DataFrame([resid(raw.iloc[i],rv.iloc[i]) for i in range(len(r))],index=r.index)
lib={'risk_adjusted_trend':pd.DataFrame({a:(p[a]/p[a].shift(20)-1)/p[a].pct_change().rolling(20,min_periods=15).std() for a in A}),'ravmom':pd.DataFrame({a:(p[a]/p[a].shift(20)-1)/p[a].pct_change().rolling(20,min_periods=15).std() for a in A}),'volnorm_reversal':pd.DataFrame({a:-(p[a]/p[a].shift(5)-1)/p[a].pct_change().rolling(5,min_periods=4).std() for a in A}),'relative_volume':pd.DataFrame({a:np.log(v[a]/v[a].rolling(20,min_periods=15).mean()) for a in A}),'vol_of_vol':pd.DataFrame({a:p[a].pct_change().rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/p[a].pct_change().rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean() for a in A}),'vix_resilience':vixres}
def test(h):
 y=pd.DataFrame({a:p[a].shift(-h)/p[a]-1 for a in A}).reindex(r.index); L=[];cov=[]
 for i in range(len(r)):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:L.append((i,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 ic=pd.Series(dict(L)); sd=ic.std(ddof=1);rank=f.rank(axis=1,pct=True);to=[]
 for i in range(1,len(r)):
  z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
  if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return ic,{'daily_paper_ic':ic.mean(),'daily_paper_icir':ic.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(ic)),'ic_hit_ratio':(ic>0).mean(),'ic_dates':len(ic),'mean_valid_instruments_per_ic_date':np.mean(cov)*15,'mean_cross_sectional_coverage':np.mean(cov),'mean_rank_turnover':np.mean(to)}
print('FACTOR diversification_correlation_20obs; PERIOD',r.index.min().date(),r.index.max().date(),'UNIVERSE',len(A)); M={}
for h in [1,5,10,20]:
 ic,m=test(h);M[h]=m;print('HORIZON',h,json.dumps(m,default=float));dates=r.index[ic.index]
 for n,mask in [('2020',dates<'2021-01-01'),('2021_2022',(dates>='2021-01-01')&(dates<'2023-01-01')),('2023_2024',(dates>='2023-01-01')&(dates<'2025-01-01')),('2025_2026',dates>='2025-01-01')]:
  q=ic[np.asarray(mask)];print('REGIME',h,n,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('SIGNAL_COVERAGE',f.notna().mean().mean(),'cells',f.notna().sum().sum(),'of',f.size);mx=0
for n,x in lib.items():
 z=pd.concat([f.stack().rename('new'),x.stack().rename('old')],axis=1).dropna();rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',rho,'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',mx,'library_files',len(glob.glob('factors/*.json')));print('DECAY',json.dumps({str(h):m for h,m in M.items()},default=float))
