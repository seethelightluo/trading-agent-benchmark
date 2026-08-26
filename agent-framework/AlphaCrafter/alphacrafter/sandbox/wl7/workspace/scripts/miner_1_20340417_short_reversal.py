import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-04-15')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in U:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  D[a]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
p=pd.DataFrame(D).sort_index().loc[:cut]; ret=p.pct_change()
# Short-horizon reversal, normalized by recent risk and lagged one completed session.
sig=(-(p/p.shift(5)-1)/(ret.rolling(60).std()*np.sqrt(5)+1e-12)).shift(1)
for label,start in [('full',pd.Timestamp('2020-01-01')),('recent750',p.index[-751]),('recent500',p.index[-501]),('recent250',p.index[-251])]:
 x=p.loc[p.index>=start]; s=sig.reindex(x.index); vals=[]; ns=[]
 for i in range(len(x)-10):
  z=pd.concat([s.iloc[i].rename('f'),(x.iloc[i+10]/x.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
 q=pd.Series(vals); print(label,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
# decay at multiple forward horizons
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic)
 q=pd.Series(vals); print('decay',h,'IC %.6f ICIR %.6f dates %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
rank=sig.rank(axis=1,pct=True)
print('period',p.index.min().date(),p.index.max().date(),'rows',len(p),'assets',len(p.columns),'coverage %.4f turnover %.4f'%(sig.notna().mean().mean(),rank.diff().abs().mean(axis=1).dropna().mean()))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20340417_short_reversal_signal.csv',index=False)
