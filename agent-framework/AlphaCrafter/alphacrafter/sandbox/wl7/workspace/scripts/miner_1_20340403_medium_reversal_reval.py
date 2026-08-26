import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-04-02')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in U:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index(); D[a]=x
p=pd.DataFrame(D).sort_index(); p=p[p.index<=cut]; r=p.pct_change()
# Revalidate the prior successful idea with the same construction, lagged one session.
f=(-(p/p.shift(20)-1)/(r.rolling(90).std()*np.sqrt(20)+1e-12)).shift(1)
for label, start in [('full',pd.Timestamp('2020-01-01')),('recent500',p.index[-501]),('recent250',p.index[-251])]:
 pp=p[p.index>=start]; ff=f.reindex(pp.index); vals=[]; ns=[]
 for i in range(len(pp)-10):
  z=pd.concat([ff.iloc[i],(pp.iloc[i+10]/pp.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.y).statistic); ns.append(len(z))
 q=pd.Series(vals).dropna(); print(label,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(q.mean(),q.mean()/q.std(ddof=1), (q>0).mean(),len(q),np.mean(ns)))
rank=f.rank(axis=1,pct=True); print('period',p.index.min().date(),p.index.max().date(),'rows',len(p),'assets',len(p.columns),'coverage %.4f turnover %.4f'%(f.notna().mean().mean(),rank.diff().abs().mean(axis=1).dropna().mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20340403_medium_reversal_reval_signal.csv',index=False)
