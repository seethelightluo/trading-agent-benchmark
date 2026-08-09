"""miner_1 20261119: residualized USD-shock resilience, one candidate factor."""
import json, glob, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-11-18'
p={}; v={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 p[a]=d.close.astype(float); v[a]=d.volume.astype(float)
r=pd.DataFrame({a:p[a].pct_change() for a in A}).sort_index()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change()
# Candidate: negative trailing DXY beta, orthogonalized daily cross-sectionally to own volatility.
def beta(x):
 z=pd.concat([x.rename('x'),dxy.rename('d')],axis=1).dropna()
 return z.x.rolling(20,min_periods=15).cov(z.d)/z.d.rolling(20,min_periods=15).var()
raw=pd.DataFrame({a:-beta(r[a]) for a in A}).reindex(r.index)
rv=pd.DataFrame({a:r[a].rolling(20,min_periods=15).std() for a in A})
def resid(y,x):
 z=pd.concat([y.rename('y'),x.rename('x')],axis=1).dropna(); out=pd.Series(np.nan,index=A)
 if len(z)>=8 and z.x.std()>0:
  X=np.c_[np.ones(len(z)),z.x];out.loc[z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
f=pd.DataFrame([resid(raw.iloc[i],rv.iloc[i]) for i in range(len(r))],index=r.index)
# Reconstruct admitted signals exactly enough for independence screen.
def vixres():
 vx=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change()
 def b(x):
  z=pd.concat([x.rename('x'),vx.rename('v')],axis=1).dropna();return z.x.rolling(20,min_periods=15).cov(z.v)/z.v.rolling(20,min_periods=15).var()
 q=pd.DataFrame({a:-b(r[a]) for a in A}).reindex(r.index)
 return pd.DataFrame([resid(q.iloc[i],rv.iloc[i]) for i in range(len(r))],index=r.index)
vol5=pd.DataFrame({a:r[a].rolling(5,min_periods=4).std() for a in A})
lib={
 'risk_adjusted_trend_20d':pd.DataFrame({a:(p[a]/p[a].shift(20)-1)/r[a].rolling(20,min_periods=15).std() for a in A}),
 'ravmom_20obs':pd.DataFrame({a:(p[a]/p[a].shift(20)-1)/r[a].rolling(20,min_periods=15).std() for a in A}),
 'volnorm_reversal_5obs':pd.DataFrame({a:-(p[a]/p[a].shift(5)-1)/r[a].rolling(5,min_periods=4).std() for a in A}),
 'relative_volume_participation_20d':pd.DataFrame({a:np.log(v[a]/v[a].rolling(20,min_periods=15).mean()) for a in A}),
 'vol_of_vol_cv20':vol5.rolling(20,min_periods=15).std()/vol5.rolling(20,min_periods=15).mean(),
 'residualized_vix_stress_resilience_beta20':vixres()}
def evaluate(h):
 y=pd.DataFrame({a:p[a].shift(-h)/p[a]-1 for a in A}).reindex(r.index); pairs=[]; cov=[]
 for i in range(len(r)):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8: pairs.append((i,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 ic=pd.Series(dict(pairs)); sd=ic.std(ddof=1)
 ranks=f.rank(axis=1,pct=True); to=[]
 for i in range(1,len(r)):
  z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
  if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 m={'daily_paper_ic':ic.mean(),'daily_paper_icir':ic.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(ic)),'ic_hit_ratio':(ic>0).mean(),'ic_dates':len(ic),'mean_valid_instruments_per_ic_date':np.mean(cov)*15,'mean_cross_sectional_coverage':np.mean(cov),'mean_rank_turnover':np.mean(to)}
 return ic,m
print('FACTOR residualized_usd_shock_resilience_beta20; PERIOD',r.index.min().date(),r.index.max().date(),'UNIVERSE',len(A))
allm={}
for h in [1,5,10,20]:
 ic,m=evaluate(h);allm[h]=m;print('HORIZON',h,json.dumps(m,default=float))
 dates=r.index[ic.index]
 for n,mask in [('2020',dates<'2021-01-01'),('2021_2022',(dates>='2021-01-01')&(dates<'2023-01-01')),('2023_2024',(dates>='2023-01-01')&(dates<'2025-01-01')),('2025_2026',dates>='2025-01-01')]:
  q=ic[np.asarray(mask)];print('REGIME',h,n,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('SIGNAL_COVERAGE',f.notna().mean().mean(),'cells',int(f.notna().sum().sum()),'of',f.size)
mx=0
for n,x in lib.items():
 z=pd.concat([f.stack().rename('new'),x.stack().rename('old')],axis=1).dropna(); rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',mx,'library_files',len(glob.glob('factors/*.json')))
print('DECAY',json.dumps({str(k):v for k,v in allm.items()},default=float))
