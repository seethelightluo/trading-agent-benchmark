import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-08-26'
def load(s):
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); return x.set_index('date')
def make(h):
 out=[]
 for s in U:
  x=load(s); r=x.close.pct_change(); f=r.rolling(20,min_periods=15).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-12); y=x.close.shift(-h)/x.close-1
  out.append(pd.DataFrame({'date':x.index,'symbol':s,'signal':f.to_numpy(),'fwd':y.to_numpy()}))
 return pd.concat(out,ignore_index=True).dropna()
def evaluate(a):
 vals=[]; ns=[]
 for d,g in a.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1:
   c=spearmanr(g.signal,g.fwd).statistic
   if pd.notna(c): vals.append((d,c)); ns.append(len(g))
 z=pd.DataFrame(vals,columns=['date','ic']).set_index('date'); return z,ns

a=make(1); z,ns=evaluate(a); q=z.ic
r=a.assign(rank=a.groupby('date').signal.rank(pct=True)).pivot(index='date',columns='symbol',values='rank')
print('trend_persistence_20d_volnorm cutoff',cut,'ic_dates',len(q),'avg_n',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',r.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-08-26')]:
 v=z.loc[lo:hi].ic; print('regime',lo,hi,'n',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
a.to_csv('scripts/miner_1_20260827_trend_persistence_20d_volnorm_signal.csv',index=False)
for h in [3,5,10]:
 zz,nn=evaluate(make(h)); print('decay',h,'n',len(zz),'IC',zz.ic.mean(),'ICIR',zz.ic.mean()/zz.ic.std(ddof=1))
print('coverage',len(a)/(sum(len(load(s)) for s in U)))
