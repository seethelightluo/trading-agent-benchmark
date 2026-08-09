import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 prev=d.close.shift(1); gap=d.open/prev-1; vol=d.close.pct_change().rolling(20).std()
 d['f']=-gap/(vol+1e-8)
 for h in [1,5,10]: d['y'+str(h)]=d.close.shift(-h)/d.close-1
 rows.append(d[['date','f','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); print('candidate normalized overnight-gap reversal')
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['f','y'+str(h)])
  if len(g)>=8:a.append((dt,spearmanr(g.f,g['y'+str(h)]).statistic,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']).set_index('date'); q=z.ic
 print(h,'dates',len(q),'avgN',z.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 if h==1:
  for yr,g in q.groupby(q.index.year): print('yr',yr,'n',len(g),'IC',g.mean(),'ICIR',g.mean()/g.std(ddof=1))
r=x.dropna(subset=['f']); print('coverage',len(r)/(len(x)),'period',x.date.min(),x.date.max())
rank=r.pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True)
print('turnover',rank.diff().abs().mean(axis=1).mean())
