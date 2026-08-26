import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2031-04-21')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]; c=x.close.astype(float); r=c.pct_change();
 D[s]=pd.DataFrame({'r3':c.pct_change(3),'v':r.rolling(20).std(),'fwd':r.shift(-1)})
common=sorted(set.intersection(*[set(v.index) for v in D.values()]))
rows=[]; ns=[]
for dt in common:
 z=pd.DataFrame({s:{k:D[s].loc[dt,k] for k in ['r3','v','fwd']} for s in U}).T.dropna()
 if len(z)>=8:
  disp=np.mean(abs(z.r3-z.r3.median()))
  if disp>z.r3.abs().median():
   f=-(z.r3-z.r3.median())/(z.v*np.sqrt(3)); q=pd.DataFrame({'f':f,'y':z.fwd}).dropna()
   if len(q)>=8: rows.append((dt,spearmanr(q.f,q.y).statistic)); ns.append(len(q))
a=pd.Series(dict(rows)).dropna(); print('dates',len(a),'avgN',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
for h in [1,5,10,20]:
 q=[]
 for dt in common:
  z=pd.DataFrame({s:{'r3':D[s].loc[dt,'r3'],'v':D[s].loc[dt,'v'],'y':D[s].fwd.rolling(h).sum().shift(-(h-1)).loc[dt]} for s in U}).T.dropna()
  if len(z)>=8 and np.mean(abs(z.r3-z.r3.median()))>z.r3.abs().median(): q.append(spearmanr(-(z.r3-z.r3.median())/(z.v*np.sqrt(3)),z.y).statistic)
 print('h',h,'IC %.8f n %d'%(np.mean(q),len(q)))
print('coverage %.6f period %s %s'%(len(a)/len(common),common[0],common[-1]))
