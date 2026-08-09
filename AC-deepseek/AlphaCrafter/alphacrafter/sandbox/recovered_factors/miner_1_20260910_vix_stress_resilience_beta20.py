"""miner_1: VIX-stress beta factor. Higher values mean lower/negative 20d sensitivity to VIX shocks."""
import json, glob
import numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-09-09'
price={}; vol={}; fw={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 price[a]=d.close.astype(float);vol[a]=d.volume.astype(float)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change()
r=pd.DataFrame({a:x.pct_change() for a,x in price.items()}).sort_index()
# Negative rolling beta to VIX: a cross-asset defensive/stress-resilience score, calculated strictly from trailing completed data.
def beta_to_vix(x):
 both=pd.concat([x.rename('x'),vix.rename('v')],axis=1).dropna()
 return both.x.rolling(20,min_periods=15).cov(both.v)/both.v.rolling(20,min_periods=15).var()
f=pd.DataFrame({a:-beta_to_vix(r[a]) for a in A}).reindex(r.index)
# All admitted-factor signal implementations, for mandatory panel Spearman independence check.
lib={}
lib['miner_3_risk_adjusted_trend_20d']=pd.DataFrame({a:(price[a]/price[a].shift(20)-1)/r[a].rolling(20,min_periods=15).std() for a in A})
lib['miner_1_ravmom_20obs']=lib['miner_3_risk_adjusted_trend_20d'].copy()
lib['miner_1_volnorm_reversal_5obs']=pd.DataFrame({a:-(price[a]/price[a].shift(5)-1)/r[a].rolling(5,min_periods=4).std() for a in A})
lib['miner_2_realized_volatility_20obs']=r.rolling(20,min_periods=15).std()
lib['miner_3_relative_volume_participation_20d']=pd.DataFrame({a:np.log(vol[a]/vol[a].rolling(20,min_periods=15).mean()) for a in A})
shortstd=r.rolling(5,min_periods=4).std(); lib['miner_1_vol_of_vol_cv20']=shortstd.rolling(20,min_periods=15).std()/shortstd.rolling(20,min_periods=15).mean()
def test(h):
 y=pd.DataFrame({a:price[a].shift(-h)/price[a]-1 for a in A}).reindex(f.index); out=[]; cover=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: out.append((dt,z.f.corr(z.y,method='spearman')));cover.append(len(z)/15)
 ic=pd.Series(dict(out));sd=ic.std(ddof=1)
 rank=f.rank(axis=1,pct=True);turn=[]
 for i in range(1,len(rank)):
  z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_std':float(sd),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_hit_ratio':float((ic>0).mean()),'ic_dates':len(ic),'mean_instruments_per_ic_date':float(np.mean(cover)*15),'mean_cross_sectional_coverage':float(np.mean(cover)),'mean_rank_turnover':float(np.mean(turn))}
print('FACTOR vix_stress_resilience_beta20; negative 20d rolling beta(asset return,VIX return)');print('PERIOD',f.index.min().date(),f.index.max().date(),'UNIVERSE',len(A))
metrics={}
for h in [1,5,10,20]:
 ic,m=test(h);metrics[h]=m;print('HORIZON',h,json.dumps(m))
 for n,mask in [('2020',ic.index<'2021-01-01'),('2021_2022',(ic.index>='2021-01-01')&(ic.index<'2023-01-01')),('2023_2024',(ic.index>='2023-01-01')&(ic.index<'2025-01-01')),('2025_2026',ic.index>='2025-01-01')]:
  q=ic[mask];print('REGIME',h,n,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('SIGNAL_COVERAGE',round(float(f.notna().mean().mean()),6),'cells',int(f.notna().sum().sum()),'of',f.size)
mx=0
for n,x in lib.items():
 z=pd.concat([f.stack().rename('new'),x.reindex(f.index).stack().rename('old')],axis=1).dropna(); rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'library_files',len(glob.glob('factors/*.json')))
print('DECAY',json.dumps({str(h):{'ic':metrics[h]['daily_paper_ic'],'icir':metrics[h]['daily_paper_icir'],'dates':metrics[h]['ic_dates']} for h in metrics}))
