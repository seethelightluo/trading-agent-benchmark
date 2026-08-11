import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def dat(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);return d.set_index('date').close.rename(s)
P=pd.concat([dat(s) for s in U],axis=1).sort_index().loc[:'2026-07-15']
R=P.pct_change(fill_method=None)
for look in [3,5,10]:
 x=R.rolling(look,min_periods=look).sum(); f=-x.sub(x.median(axis=1),axis=0)
 for h in [1,5,10]:
  y=sum(R.shift(-k) for k in range(1,h+1)); rows=[]
  for dt in R.index:
   z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
   if len(z)>=8 and z.f.nunique()>1: rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
  q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');o=q.ic
  if len(o): print('look',look,'h',h,'dates',len(o),'avgN',q.n.mean(),'IC',o.mean(),'ICIR',o.mean()/o.std(),'hit',(o>0).mean())
  else: print('look',look,'h',h,'NO DATA')
