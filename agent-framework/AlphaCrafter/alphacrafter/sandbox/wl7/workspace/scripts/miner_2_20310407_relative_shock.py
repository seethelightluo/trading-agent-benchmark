import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); c=x.close.astype(float); r=c.pct_change()
 D[s]=pd.DataFrame({'r3':c.pct_change(3),'vol':r.rolling(20).std(),'fwd':r.shift(-1)})
common=sorted(set.intersection(*[set(v.index) for v in D.values()]))
ics=[]; ns=[]
for dt in common:
 z=pd.DataFrame({s:{k:D[s].loc[dt,k] for k in ['r3','vol','fwd']} for s in U}).T.dropna()
 if len(z)>=8:
  # relative shock, contrarian signal
  f=-(z.r3-z.r3.median())/(z.vol*np.sqrt(3))
  ics.append(spearmanr(f,z.fwd).statistic); ns.append(len(z))
a=pd.Series(ics)
print('dates',len(a),'avgN',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
for h in [1,5,10,20]:
 q=[]
 for dt in common:
  z=pd.DataFrame({s:{'r3':D[s].loc[dt,'r3'],'vol':D[s].loc[dt,'vol'],'fwd':D[s]['fwd'].shift(-(h-1)).loc[dt]} for s in U}).T.dropna()
  if len(z)>=8:q.append(spearmanr(-(z.r3-z.r3.median())/(z.vol*np.sqrt(3)),z.fwd).statistic)
 print('h',h,'IC %.8f n %d'%(np.mean(q),len(q)))
print('coverage',len(a)/len(common),'period',common[0],common[-1])
