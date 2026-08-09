"""miner_1 candidate: residualized upside-VIX shock resilience, validated as of 2026-10-07."""
import json,glob
import numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-10-07'
price={}; volume={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 price[a]=d.close.astype(float);volume[a]=d.volume.astype(float)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change()
r=pd.DataFrame({a:x.pct_change() for a,x in price.items()}).sort_index()
# Only positive VIX-return observations enter covariance; score is negative conditional beta.
def shockbeta(x):
 z=pd.concat([x.rename('asset'),vix.rename('vix')],axis=1).dropna(); z.loc[z.vix<=0,['asset','vix']]=np.nan
 return z.asset.rolling(30,min_periods=12).cov(z.vix)/z.vix.rolling(30,min_periods=12).var()
raw=pd.DataFrame({a:-shockbeta(r[a]) for a in A}).reindex(r.index)
rv=pd.DataFrame({a:price[a].pct_change().rolling(20,min_periods=15).std() for a in A}).reindex(r.index)
def resid(y,x):
 z=pd.concat([y.rename('y'),x.rename('x')],axis=1).dropna();o=pd.Series(np.nan,index=A)
 if len(z)>=8 and z.x.std()>0:
  X=np.c_[np.ones(len(z)),z.x];o.loc[z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
f=pd.DataFrame([resid(raw.iloc[i],rv.iloc[i]) for i in range(len(r))],index=r.index)
# Reconstruct every admitted signal exactly enough to perform mandatory pooled-cell Spearman screen.
lib={
'miner_3_risk_adjusted_trend_20d':pd.DataFrame({a:(price[a]/price[a].shift(20)-1)/price[a].pct_change().rolling(20,min_periods=15).std() for a in A}).reindex(r.index),
'miner_1_ravmom_20obs':pd.DataFrame({a:(price[a]/price[a].shift(20)-1)/price[a].pct_change().rolling(20,min_periods=15).std() for a in A}).reindex(r.index),
'miner_1_volnorm_reversal_5obs':pd.DataFrame({a:-(price[a]/price[a].shift(5)-1)/price[a].pct_change().rolling(5,min_periods=4).std() for a in A}).reindex(r.index),
'miner_2_realized_volatility_20obs':rv,
'miner_3_relative_volume_participation_20d':pd.DataFrame({a:np.log(volume[a]/volume[a].rolling(20,min_periods=15).mean()) for a in A}).reindex(r.index),
'miner_1_vol_of_vol_cv20':pd.DataFrame({a:price[a].pct_change().rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/price[a].pct_change().rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean() for a in A}).reindex(r.index),
'miner_1_residualized_vix_stress_resilience_beta20':None}
# Existing factor calculated using all VIX days, residualized on rv.
def fullbeta(x):
 z=pd.concat([x.rename('x'),vix.rename('v')],axis=1).dropna();return z.x.rolling(20,min_periods=15).cov(z.v)/z.v.rolling(20,min_periods=15).var()
oldraw=pd.DataFrame({a:-fullbeta(r[a]) for a in A}).reindex(r.index)
lib['miner_1_residualized_vix_stress_resilience_beta20']=pd.DataFrame([resid(oldraw.iloc[i],rv.iloc[i]) for i in range(len(r))],index=r.index)
def evaluate(h):
 y=pd.DataFrame({a:price[a].shift(-h)/price[a]-1 for a in A}).reindex(r.index); obs=[];coverage=[]
 for i in range(len(r)):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:obs.append((i,z.f.corr(z.y,method='spearman')));coverage.append(len(z)/15)
 ic=pd.Series(dict(obs)); ranks=f.rank(axis=1,pct=True);to=[]
 for i in range(1,len(r)):
  z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
  if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 sd=ic.std(ddof=1);return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_std':float(sd),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_hit_ratio':float((ic>0).mean()),'ic_dates':len(ic),'mean_valid_instruments_per_ic_date':float(np.mean(coverage)*15),'mean_cross_sectional_coverage':float(np.mean(coverage)),'mean_rank_turnover':float(np.mean(to))}
print('FACTOR upside_vix_shock_resilience_residual_beta30; negative 30-observation beta only on positive VIX-return days, cross-sectionally residualized against 20-observation realized volatility')
print('PERIOD',r.index.min().date(),r.index.max().date(),'UNIVERSE',len(A)); M={}
for h in [1,5,10,20]:
 ic,m=evaluate(h);M[h]=m;print('HORIZON',h,json.dumps(m));ds=r.index[ic.index]
 for name,mask in [('2020',ds<'2021-01-01'),('2021_2022',(ds>='2021-01-01')&(ds<'2023-01-01')),('2023_2024',(ds>='2023-01-01')&(ds<'2025-01-01')),('2025_2026',ds>='2025-01-01')]:
  q=ic[np.asarray(mask)];print('REGIME',h,name,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('SIGNAL_COVERAGE',round(float(f.notna().mean().mean()),6),'cells',int(f.notna().sum().sum()),'of',f.size)
mx=0
for n,x in lib.items():
 z=pd.concat([f.stack().rename('new'),x.stack().rename('old')],axis=1).dropna();rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'library_files',len(glob.glob('factors/*.json')))
print('DECAY',json.dumps({str(h):{'ic':M[h]['daily_paper_ic'],'icir':M[h]['daily_paper_icir'],'dates':M[h]['ic_dates']} for h in M}))
