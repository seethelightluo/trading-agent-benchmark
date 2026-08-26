import pandas as pd, numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-10-02')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in syms}).sort_index().loc[:cut]
r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
# smooth multi-horizon reversal: recent 5d reversal plus 20d reversal, normalized by risk
sig=-(p.pct_change(5)+0.5*p.pct_change(20))/vol
out=[]; art=[]
for d in sig.index:
 f=p.shift(-10).loc[d]/p.loc[d]-1; z=pd.concat([sig.loc[d].rename('x'),f.rename('y')],axis=1).dropna()
 if len(z)>=8:
  out.append((d,len(z),spearmanr(z.x,z.y).statistic))
  for s in z.index: art.append((d,s,float(sig.loc[d,s]),float(f[s])))
a=pd.DataFrame(out,columns=['date','n','ic']).set_index('date')
print('candidate=blended_volnorm_reversal cutoff=2034-10-02'); print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for x,y in [('2023','2026'),('2027','2030'),('2031','2034')]:
 q=a.loc[x:y].ic; print(x,y,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for h in [1,5,10,20]:
 v=[]
 for d in sig.index:
  f=p.shift(-h).loc[d]/p.loc[d]-1; z=pd.concat([sig.loc[d].rename('x'),f.rename('y')],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.x,z.y).statistic)
 print('horizon',h,'dates',len(v),'IC',np.mean(v),'ICIR',np.mean(v)/np.std(v,ddof=1))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
pd.DataFrame(art,columns=['date','symbol','signal','forward_10d_return']).to_csv('scripts/miner_1_20341002_blended_reversal_signal.csv',index=False)
print('artifact rows',len(art))
