"""miner_2: validate one interpretable conditional high-beta trend factor through prior completed date."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
E=pd.Timestamp('2026-09-09')
P,V={},{}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:E]
 P[a]=d.close.astype(float); V[a]=d.volume.astype(float)
p=pd.DataFrame(P).sort_index(); v=pd.DataFrame(V).sort_index(); r=p.pct_change(fill_method=None)
# One idea: assets with high common-risk sensitivity only receive a positive score when their own 20-observation trend is positive.
# Market proxy is equal-weight available same-day cross-asset return, using only tradable data.
mkt=r.mean(axis=1,skipna=True)
beta=pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(mkt)/mkt.rolling(20,min_periods=15).var() for a in A})
trend=p/p.shift(20)-1
f=beta*np.sign(trend) # conditional high-beta participation, not a raw momentum magnitude signal
lib={
 'miner_3_risk_adjusted_trend_20d':trend/r.rolling(20,min_periods=15).std(),
 'miner_1_ravmom_20obs':trend/r.rolling(20,min_periods=15).std(),
 'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),
 'miner_3_relative_volume_participation_20d':np.log(v/v.rolling(20,min_periods=15).mean()),
 'miner_1_vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean()
}
def evaluate(h):
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:
   vals.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)); ns.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s, {'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_std':float(sd),'ic_hit_ratio':float((s>0).mean()),'n_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
M={}
for h in (1,5,10,20):
 s,m=evaluate(h);M[h]=m; print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for name,mask in [('2020',s.index<'2021-01-01'),('2021_22',(s.index>='2021-01-01')&(s.index<'2023-01-01')),('2023_24',(s.index>='2023-01-01')&(s.index<'2025-01-01')),('2025_26',s.index>='2025-01-01'),('latest_90d',s.index>=E-pd.Timedelta(days=90))]:
   q=s[mask]; print('REGIME',name,'n',len(q),'IC',float(q.mean()),'ICIR',float(q.mean()/q.std(ddof=1)))
t=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
C={}
for name,x in lib.items():
 z=pd.concat([f.stack().rename('a'),x.stack().rename('b')],axis=1).dropna(); C[name]=float(spearmanr(z.a,z.b).statistic);print('LIBRARY_CORR',name,'pairs',len(z),'rho',C[name])
print('FACTOR conditional_high_beta_trend_20obs')
print('PERIOD',f.index.min().date(),E.date(),'panel_dates',len(f),'cell_coverage',float(f.notna().mean().mean()),'mean_names',float(f.notna().sum(axis=1).mean()),'mean_rank_turnover',float(np.mean(t)))
print('DECAY',json.dumps({str(h):M[h] for h in M},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',max(map(abs,C.values())))
