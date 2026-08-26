import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1)
print('data',p.index.min().date(),p.index.max().date(),'assets',len(D))
# Market-neutral momentum: 20d asset return less its rolling-beta exposure to common cross-asset return.
def run(h):
 rows=[]
 for i,t in enumerate(p.index):
  if i<65 or i+h>=len(p): continue
  rr=r.iloc[i-59:i+1]; mm=m.iloc[i-59:i+1]
  beta=rr.apply(lambda x: x.cov(mm)/(mm.var()+1e-12))
  ar=p.iloc[i]/p.iloc[i-20]-1
  sig=ar-beta*(p.iloc[i]/p.iloc[i-20]-1).mean()
  f=p.iloc[i+h]/p.iloc[i]-1
  q=pd.concat([sig,f],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
 A=pd.DataFrame(rows,columns=['date','n','ic'])
 print('H',h,'dates',len(A),'mean_n',round(A.n.mean(),2),'coverage',round(A.n.mean()/15,4))
 for label,cond in [('full',A.date>=A.date.min()),('2020_23',(A.date>='2020-01-01')&(A.date<'2024-01-01')),('2024_26',(A.date>='2024-01-01')&(A.date<'2027-01-01')),('2027_28',(A.date>='2027-01-01')&(A.date<'2029-01-01')),('recent',(A.date>='2028-05-01'))]:
  q=A[cond].ic
  print(label,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
for h in [5,10,20]: run(h)
