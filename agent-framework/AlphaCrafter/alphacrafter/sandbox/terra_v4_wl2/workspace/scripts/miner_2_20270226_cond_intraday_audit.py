import os,json,glob
import pandas as pd,numpy as np
from scipy.stats import spearmanr
cand='../persistent/factor_signals_miner_2_20270225_cond_intraday1.csv'
A=pd.read_csv(cand).rename(columns={'asset':'symbol'}).set_index(['date','symbol']).signal
# forward returns from local data
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}
p=pd.concat(px,axis=1); y=(p.shift(-1)/p-1).stack(); y.index.names=['date','symbol']
def stats(mask):
 z=[]; ns=[]
 for dt,g in A[mask].groupby(level=0):
  q=pd.concat([g,y[g.index]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z)
 return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)),float((z>0).mean())
print('ALL',stats(A.notna()))
ix=A.index.get_level_values(0)
print('2027',stats((ix>='2027-01-01')&(ix<='2027-02-25')))
print('2026',stats((ix>='2026-01-01')&(ix<='2026-12-31')))
print('pre2026',stats(ix<'2026-01-01'))
# audit all json artifact paths
res=[]
for root,ds,fs in os.walk('factors'):
 for fn in fs:
  if not fn.endswith('.json'): continue
  try:x=json.load(open(os.path.join(root,fn)))
  except:continue
  prov=x.get('provenance',{}); path=prov.get('signal_artifact') or x.get('signal_artifact') or x.get('signal_provenance',{}).get('artifact')
  if not path:continue
  if path.startswith('../'): fp=path
  else: fp=path
  if not os.path.exists(fp):continue
  try:
   d=pd.read_csv(fp); sym='symbol' if 'symbol' in d else ('asset' if 'asset' in d else None)
   if not sym or 'signal' not in d:continue
   B=d.rename(columns={sym:'symbol'}).set_index(['date','symbol']).signal
   q=pd.concat([A,B],axis=1).dropna();
   if len(q)>30:
    res.append((fn,len(q),q.iloc[:,0].corr(q.iloc[:,1]),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
  except Exception as e: pass
res.sort(key=lambda x:abs(x[2]),reverse=True)
print('AUDIT',len(res))
for r in res[:15]:print(r)
