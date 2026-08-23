import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-05-26')
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index().loc[:CUT]
# Cross-sectional standardized agreement: mean of rank-normalized 5/20/60-day returns, rewarding aligned trend.
rets={h:P/P.shift(h)-1 for h in [5,20,60]}
f=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
for dt in P.index:
    z=[]
    for h in [5,20,60]: z.append(rets[h].loc[dt].rank(pct=True))
    f.loc[dt]=pd.concat(z,axis=1).mean(axis=1)
fr={h:P.shift(-h)/P-1 for h in [5,10,20]}
print('cutoff',P.index.max().date(),'dates',len(P),'assets',len(A),'coverage',round(f.notna().stack().mean(),6))
for h in [5,10,20]:
 ic=[];ns=[];turn=[]; prev=None
 for dt in P.index:
  z=pd.concat([f.loc[dt].rename('x'),fr[h].loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.x,z.y).statistic);ns.append(len(z))
  if prev is not None:
   q=pd.concat([f.loc[dt].rank(pct=True),prev],axis=1).dropna()
   if len(q):turn.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
  prev=f.loc[dt].rank(pct=True)
 x=np.array(ic); print('horizon',h,'valid_dates',len(x),'avgN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(x.mean(),8),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(len(x)),6),'hit',round(np.mean(x>0),5),'turnover',round(np.mean(turn),6))
 if h==10:
  for yr in range(2020,2033):
   q=[ic[j] for j,dt in enumerate([d for d in P.index if len(pd.concat([f.loc[d].rename('x'),fr[h].loc[d].rename('y')],axis=1).dropna())>=8]) if dt.year==yr]
   if q: print('regime',yr,len(q),round(float(np.mean(q)),6))
