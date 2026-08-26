import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1)
print('data',p.index.min().date(),p.index.max().date(),'assets',len(D))
rows=[]
for i,t in enumerate(p.index):
 if i<65 or i+5>=len(p): continue
 rr=r.iloc[i-59:i+1]; mm=m.iloc[i-59:i+1]; beta=rr.apply(lambda x:x.cov(mm)/(mm.var()+1e-12))
 ret5=p.iloc[i]/p.iloc[i-5]-1; residual=ret5-beta*ret5.mean(); vol=rr.iloc[-10:].std()+1e-8
 sig=-residual/vol; f=p.iloc[i+5]/p.iloc[i]-1
 q=pd.concat([sig,f],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(rows,columns=['date','n','ic']); print('dates',len(A),'mean_n',A.n.mean(),'coverage',A.n.mean()/15)
for lab,c in [('full',A.date>=A.date.min()),('2020_23',(A.date<'2024-01-01')),('2024_26',(A.date>='2024-01-01')&(A.date<'2027-01-01')),('2027_28',(A.date>='2027-01-01')&(A.date<'2029-01-01')),('recent',(A.date>='2028-05-01'))]:
 q=A[c].ic; print(lab,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
