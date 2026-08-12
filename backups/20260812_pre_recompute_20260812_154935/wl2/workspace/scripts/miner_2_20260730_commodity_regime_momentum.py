import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.concat(D,axis=1).sort_index(); r=P.pct_change(20); macro=(r.XAU-r.COPPER).shift(1); sig=r.mul(np.sign(macro),axis=0).shift(1); fwd=P.pct_change().shift(-1)
def calc(ret):
 out=[]; ns=[]
 for dt in sig.index:
  a=sig.loc[dt].dropna(); b=ret.loc[dt].dropna(); ix=a.index.intersection(b.index); a=a[ix]; b=b[ix]
  if len(ix)>=8 and a.nunique()>1 and b.nunique()>1: out.append((dt,spearmanr(a,b).statistic)); ns.append(len(ix))
 return pd.DataFrame(out,columns=['date','ic']).set_index('date'),ns
q,ns=calc(fwd)
for label,z in [('all',q),('2020-22',q.loc['2020':'2022']),('2023-24',q.loc['2023':'2024']),('2025-26',q.loc['2025':'2026-07-15']),('recent',q.tail(252))]:
 print(label,'dates',len(z),'IC %.6f ICIR %.6f hit %.3f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
print('avg instruments',np.mean(ns),'coverage',np.mean(ns)/15,'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[q.index].mean()/2)
for h in [5,10]:
 z,n=calc(P.pct_change(h).shift(-h)); print('h',h,'dates',len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)))
print('range',q.index.min().date(),q.index.max().date())
