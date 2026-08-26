import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); c=x.close.astype(float); r=c.pct_change()
 # Volatility-of-volatility conditioned medium momentum: trend is preferred when recent vol is stable
 v=r.rolling(20).std(); vv=v.pct_change(5).abs().rolling(10).mean()
 D[s]=pd.DataFrame({'f':c.pct_change(10)/(v*np.sqrt(10))*(1/(1+vv*10)),'r':r})
common=sorted(set.intersection(*[set(v.index) for v in D.values()]))
ics=[]; ns=[]
for dt in common:
 z=pd.DataFrame({s:{'f':D[s].loc[dt,'f'],'fr':D[s].r.shift(-1).loc[dt]} for s in U}).T.dropna()
 if len(z)>=8: ics.append(spearmanr(z.f,z.fr).statistic); ns.append(len(z))
a=pd.Series(ics)
print('dates',len(a),'avgN',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
for h in [1,5,10,20]:
 q=[]
 for dt in common:
  z=pd.DataFrame({s:{'f':D[s].loc[dt,'f'],'fr':D[s].r.rolling(h).sum().shift(-(h-1)).loc[dt]} for s in U}).T.dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.fr).statistic)
 print('h',h,'IC %.8f n %d'%(np.mean(q),len(q)))
# rank turnover
prev=None; ch=[]
for dt in common:
 z=pd.Series({s:D[s].loc[dt,'f'] for s in U}).dropna()
 if len(z)>=8:
  q=z.rank(pct=True)
  if prev is not None: ch.append(np.mean(abs(q-prev.reindex(q.index))))
  prev=q
print('coverage',len(a)/len(common),'turnover %.8f'%np.mean(ch),'period',common[0],common[-1],'avgN',np.mean(ns))
