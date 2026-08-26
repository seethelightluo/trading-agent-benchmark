import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); out={}
# Relative residual reversal: remove common cross-asset 20d move, then fade each asset's residual.
# At date t, residual is asset 20d return minus cross-sectional median 20d return.
for h in [1,5,10,20,40]:
 rows=[]
 for i,t in enumerate(p.index):
  if i<25 or i+h>=len(p): continue
  rr=p.iloc[i]/p.iloc[i-20]-1
  sig=-(rr-rr.median())
  f=p.iloc[i+h]/p.iloc[i]-1
  q=pd.concat([sig,f],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
 A=pd.DataFrame(rows,columns=['date','n','ic'])
 x=A.ic.to_numpy(); out[h]={'dates':len(A),'mean_n':float(A.n.mean()),'ic':float(x.mean()),'icir':float(x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(252)),'hit':float((x>0).mean()),'recent_ic':float(A.tail(252).ic.mean()),'recent_icir':float(A.tail(252).ic.mean()/(A.tail(252).ic.std(ddof=1)+1e-12)*np.sqrt(252))}
print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'rows',len(p))
print(out)
