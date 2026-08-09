import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill().loc[:END]
R=P.pct_change(); lag=R.shift(1)
disp=lag.rolling(20,min_periods=15).std().mean(axis=1)
med=disp.rolling(120,min_periods=60).median()
active=(disp>med).astype(float)
raw=-lag.rolling(3,min_periods=3).sum().sub(lag.rolling(3,min_periods=3).sum().median(axis=1),axis=0)
F=raw.mul(active,axis=0)
F.to_csv('scripts/miner_3_20261217_dispersion_conditioned_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; vals=[]; ns=[]
 for dt in F.index:
  q=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: vals.append(spearmanr(q.f,q.y).statistic); ns.append(len(q))
 a=np.asarray(vals); print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('active_rate',active.mean(),'coverage',F.notna().sum().sum()/F.size,'nonzero',((F!=0)&F.notna()).sum().sum()/F.notna().sum().sum())
for label,ix in [('2020_22',F.index<'2023-01-01'),('2023_24',(F.index>='2023-01-01')&(F.index<'2025-01-01')),('2025_26',F.index>='2025-01-01')]:
 Y=P.shift(-1).div(P)-1; a=[]
 for dt in F.index[ix]:
  q=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:a.append(spearmanr(q.f,q.y).statistic)
 a=np.asarray(a); print(label,len(a),a.mean(),a.mean()/a.std(ddof=1))
# provenance correlation to pre-existing artifacts, excluding this output
cs=[]
for p in Path('scripts').glob('*signal.csv'):
 if p.name.endswith('dispersion_conditioned_reversal_signal.csv'): continue
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U)
  c=F.stack().corr(x.stack())
  if pd.notna(c):cs.append((abs(c),p.name,c))
 except Exception: pass
print('max_abs_other_artifact_corr',max(cs) if cs else None)
print('period',F.index.min().date(),F.index.max().date(),'instruments',len(U))
