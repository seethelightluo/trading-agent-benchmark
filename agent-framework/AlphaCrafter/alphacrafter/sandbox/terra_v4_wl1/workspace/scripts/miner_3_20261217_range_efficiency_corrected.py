import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); r=d.close.pct_change(); d['f']=d.close.pct_change(20)/(r.abs().rolling(20).sum()+1e-12)
 for h in [1,5,10]: d['y'+str(h)]=d.close.shift(-h)/d.close-1
 rows.append(d[['date','f','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); out={}
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['f','y'+str(h)])
  if len(g)>=8:a.append((dt,spearmanr(g.f,g['y'+str(h)]).statistic,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']).set_index('date'); q=z.ic; out[h]=z
 print(h,len(q),z.n.mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
 if h==1:
  for yr,g in q.groupby(q.index.year):print('yr',yr,len(g),g.mean(),g.mean()/g.std(ddof=1))
r=x.dropna(subset=['f']); ranks=r.pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True)
print('coverage',len(r)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean())
# pooled correlation with existing plain reversal proxy
p=x.pivot(index='date',columns='symbol',values='f'); rev=-pd.concat([pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.pct_change(5).rename(s) for s in syms],axis=1)
print('corr_plain_rev',p.stack().corr(rev.reindex(p.index).stack()))
print('period',x.date.min(),x.date.max())
