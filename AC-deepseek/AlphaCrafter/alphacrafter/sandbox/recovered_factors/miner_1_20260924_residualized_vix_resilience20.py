"""miner_1: residualized VIX stress resilience. Removes each day's cross-asset realized-volatility exposure from negative 20d VIX beta."""
import json, glob
import numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-09-23'
price={}; vol={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]; price[a]=d.close.astype(float);vol[a]=d.volume.astype(float)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change()
r=pd.DataFrame({a:x.pct_change() for a,x in price.items()}).sort_index()
def b(x):
 z=pd.concat([x.rename('x'),vix.rename('v')],axis=1).dropna();return z.x.rolling(20,min_periods=15).cov(z.v)/z.v.rolling(20,min_periods=15).var()
raw=pd.DataFrame({a:-b(r[a]) for a in A}).reindex(r.index);rv=r.rolling(20,min_periods=15).std()
def resid(y,x):
 z=pd.concat([y.rename('y'),x.rename('x')],axis=1).dropna(); out=pd.Series(np.nan,index=A)
 if len(z)>=8 and z.x.std()!=0:
  X=np.column_stack([np.ones(len(z)),z.x]);out.loc[z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
# Positional construction preserves all duplicate-calendar rows present in the union cross-asset panel.
f=pd.DataFrame([resid(raw.iloc[i],rv.iloc[i]) for i in range(len(raw))],index=raw.index)
lib={'miner_3_risk_adjusted_trend_20d':pd.DataFrame({a:(price[a]/price[a].shift(20)-1)/r[a].rolling(20,min_periods=15).std() for a in A}),'miner_1_ravmom_20obs':pd.DataFrame({a:(price[a]/price[a].shift(20)-1)/r[a].rolling(20,min_periods=15).std() for a in A}),'miner_1_volnorm_reversal_5obs':pd.DataFrame({a:-(price[a]/price[a].shift(5)-1)/r[a].rolling(5,min_periods=4).std() for a in A}),'miner_2_realized_volatility_20obs':rv,'miner_3_relative_volume_participation_20d':pd.DataFrame({a:np.log(vol[a]/vol[a].rolling(20,min_periods=15).mean()) for a in A}),'miner_1_vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean()}
def test(h):
 y=pd.DataFrame({a:price[a].shift(-h)/price[a]-1 for a in A}).reindex(f.index);out=[];cv=[]
 for i,dt in enumerate(f.index):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:out.append((i,z.f.corr(z.y,method='spearman')));cv.append(len(z)/15)
 ic=pd.Series(dict(out));sd=ic.std(ddof=1);rank=f.rank(axis=1,pct=True);turn=[]
 for i in range(1,len(rank)):
  z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_std':float(sd),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_hit_ratio':float((ic>0).mean()),'ic_dates':len(ic),'mean_instruments_per_ic_date':float(np.mean(cv)*15),'mean_cross_sectional_coverage':float(np.mean(cv)),'mean_rank_turnover':float(np.mean(turn))}
print('FACTOR residualized_vix_stress_resilience_beta20; daily cross-sectional residual of negative 20d VIX beta after realized-volatility OLS');print('PERIOD',f.index.min().date(),f.index.max().date(),'UNIVERSE',len(A))
M={}
for h in [1,5,10,20]:
 ic,m=test(h);M[h]=m;print('HORIZON',h,json.dumps(m))
 # IC positional index inherits chronological order, so regime masks use source calendar positions.
 dates=f.index[ic.index]
 for n,mask in [('2020',dates<'2021-01-01'),('2021_2022',(dates>='2021-01-01')&(dates<'2023-01-01')),('2023_2024',(dates>='2023-01-01')&(dates<'2025-01-01')),('2025_2026',dates>='2025-01-01')]:
  q=ic[np.asarray(mask)];print('REGIME',h,n,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('SIGNAL_COVERAGE',round(float(f.notna().mean().mean()),6),'cells',int(f.notna().sum().sum()),'of',f.size)
mx=0; evidence={}
for n,x in lib.items():
 # row-position pairing is necessary for duplicate dates in the master calendar.
 cells=[]
 for i in range(len(f)):
  z=pd.concat([f.iloc[i].rename('new'),x.reindex(f.index).iloc[i].rename('old')],axis=1).dropna();cells.append(z)
 z=pd.concat(cells);rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));evidence[n]=(rho,len(z));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'library_files',len(glob.glob('factors/*.json')))
print('DECAY',json.dumps({str(h):{'ic':M[h]['daily_paper_ic'],'icir':M[h]['daily_paper_icir'],'dates':M[h]['ic_dates']} for h in M}))
