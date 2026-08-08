"""miner_2 one-idea test: low market-beta defensive factor, through 2026-08-12 only."""
import json,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2026-08-12')
P={};V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:E];P[a]=d.close.astype(float);V[a]=d.volume.astype(float)
# SPX is an observable common-risk proxy. On each asset's own trading calendar, forward-fill only the already-observed SPX daily return, then compute 20 *observations* beta.
spx=P['SPX'].pct_change(fill_method=None); F={}
for a,p in P.items():
 ar=p.pct_change(fill_method=None); mr=spx.reindex(ar.index,method='ffill')
 F[a]=-ar.rolling(20,min_periods=15).cov(mr)/mr.rolling(20,min_periods=15).var()
f=pd.DataFrame(F).sort_index();p=pd.DataFrame(P).sort_index();r=p.pct_change(fill_method=None)
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
lib={'miner_1_ravmom_20obs':trend,'miner_3_risk_adjusted_trend_20d':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_2_realized_volatility_20obs':-r.rolling(20,min_periods=15).std(),'miner_3_relative_volume_participation_20d':np.log(pd.DataFrame(V)/pd.DataFrame(V).rolling(20,min_periods=15).mean())}
def calc(h):
 fw=pd.DataFrame({a:P[a].shift(-h)/P[a]-1 for a in A});o=[];n=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:o.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic));n.append(len(z))
 s=pd.Series(dict(o));sd=s.std(ddof=1);return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_std':float(sd),'ic_hit_ratio':float((s>0).mean()),'n_dates':len(s),'mean_valid_instruments':float(np.mean(n))}
M={}
for h in [1,5,10,20]:
 s,m=calc(h);M[h]=m;print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for name,mask in [('2020',s.index<'2021-01-01'),('2021_22',(s.index>='2021-01-01')&(s.index<'2023-01-01')),('2023_24',(s.index>='2023-01-01')&(s.index<'2025-01-01')),('2025_26',s.index>='2025-01-01')]:
   q=s[mask];print('REGIME',name,'n',len(q),'IC',float(q.mean()),'ICIR',float(q.mean()/q.std(ddof=1)))
t=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
C={}
for name,x in lib.items():
 z=pd.concat([f.stack().rename('new'),x.stack().rename('old')],axis=1).dropna();C[name]=float(spearmanr(z.new,z.old).statistic);print('LIBRARY_CORR',name,'n_pairs',len(z),'rho',C[name])
print('FACTOR low_spx_beta_20obs');print('PERIOD',f.index.min().date(),E.date(),'panel_dates',len(f),'coverage',float(f.notna().mean().mean()),'mean_names',float(f.notna().sum(axis=1).mean()),'mean_rank_turnover_1d',float(np.mean(t)))
print('DECAY',json.dumps({str(h):{'ic':M[h]['daily_paper_ic'],'icir':M[h]['daily_paper_icir'],'n_dates':M[h]['n_dates']} for h in M}));print('MAX_ABS_LIBRARY_CORRELATION',max(abs(x) for x in C.values()))
