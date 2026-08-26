import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 c=x['close'].astype(float); r=c.pct_change()
 f=(c.pct_change(5)-c.pct_change(20)/4)/(r.rolling(20).std()*np.sqrt(20))
 D[s]=pd.DataFrame({'f':f,'r':r})
common=set.intersection(*[set(v.index) for v in D.values()])
rows=[]; ns=[]
for dt in sorted(common):
 z=pd.DataFrame({s:{'f':D[s].loc[dt,'f'],'fr':D[s]['r'].shift(-1).loc[dt]} for s in U}).T.dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.f,z.fr).statistic)); ns.append(len(z))
a=pd.Series(dict(rows)).dropna(); print('dates',len(a),'avgN',np.mean(ns))
print('IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
for h in [1,5,10,20]:
 rr=[]
 for dt in sorted(common):
  z=pd.DataFrame({s:{'f':D[s].loc[dt,'f'],'fr':D[s]['r'].rolling(h).sum().shift(-(h-1)).loc[dt]} for s in U}).T.dropna()
  if len(z)>=8: rr.append(spearmanr(z.f,z.fr).statistic)
 print('h',h,'IC %.8f n %d'%(np.nanmean(rr),len(rr)))
prev=None; changes=[]
for dt in sorted(common):
 z=pd.Series({s:D[s].loc[dt,'f'] for s in U}).dropna()
 if len(z)>=8:
  rank=z.rank(pct=True)
  if prev is not None: changes.append(np.mean(abs(rank-prev.reindex(rank.index))))
  prev=rank
print('turnover',np.mean(changes),'coverage',len(a)/len(common),'period',a.index.min(),a.index.max())
