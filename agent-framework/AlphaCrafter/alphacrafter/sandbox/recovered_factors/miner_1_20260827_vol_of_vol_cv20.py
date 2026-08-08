"""miner_1: 20d volatility-of-volatility factor; independent risk-instability candidate."""
import json,glob
import numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-08-26'
F={}; Y={}; L={k:{} for k in ['miner_3_risk_adjusted_trend_20d','miner_3_relative_volume_participation_20d','miner_1_ravmom_20obs','miner_1_volnorm_reversal_5obs','miner_2_realized_volatility_20obs']}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 p=d.close.astype(float);r=p.pct_change();v=d.volume.astype(float)
 # coefficient of variation of trailing 5d realized volatility over 20 observations; high = unstable risk
 rv=r.rolling(5,min_periods=4).std(); F[a]=rv.rolling(20,min_periods=15).std()/rv.rolling(20,min_periods=15).mean()
 for h in [1,5,10,20]:Y[a,h]=p.shift(-h)/p-1
 L['miner_3_risk_adjusted_trend_20d'][a]=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
 L['miner_1_ravmom_20obs'][a]=L['miner_3_risk_adjusted_trend_20d'][a]
 L['miner_1_volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
 L['miner_2_realized_volatility_20obs'][a]=r.rolling(20,min_periods=15).std()
 L['miner_3_relative_volume_participation_20d'][a]=np.log(v/v.rolling(20,min_periods=15).mean())
f=pd.DataFrame(F).sort_index()
def ev(h):
 y=pd.DataFrame({a:Y[a,h] for a in A}).sort_index();zout=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:zout.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 ic=pd.Series(dict(zout));sd=ic.std(ddof=1);rk=f.rank(axis=1,pct=True);to=[]
 for i in range(1,len(rk)):
  z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
  if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_std':float(sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_cross_sectional_coverage':float(np.mean(cov)),'mean_rank_turnover':float(np.mean(to))}
print('FACTOR vol_of_vol_cv20 = std(rolling_std(return,5),20)/mean(rolling_std(return,5),20)');print('VALIDATION_CUTOFF',END,'instruments',len(A),'signal_dates',len(f))
M={}
for h in [1,5,10,20]:
 ic,m=ev(h);M[h]=m;print('HORIZON',h,json.dumps(m))
 for label,mask in [('2020',ic.index<'2021-01-01'),('2021_22',(ic.index>='2021-01-01')&(ic.index<'2023-01-01')),('2023_24',(ic.index>='2023-01-01')&(ic.index<'2025-01-01')),('2025_26',ic.index>='2025-01-01')]:
  x=ic[mask];print('REGIME',h,label,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('CELL_COVERAGE',round(float(f.notna().mean().mean()),6))
mx=0
for n,x in L.items():
 z=pd.concat([f.stack().rename('new'),pd.DataFrame(x).stack().rename('old')],axis=1).dropna();rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'library_files',len(glob.glob('factors/*.json')))
print('DECAY',json.dumps({str(h):{'ic':M[h]['daily_paper_ic'],'icir':M[h]['daily_paper_icir'],'dates':M[h]['ic_dates']} for h in M}))
