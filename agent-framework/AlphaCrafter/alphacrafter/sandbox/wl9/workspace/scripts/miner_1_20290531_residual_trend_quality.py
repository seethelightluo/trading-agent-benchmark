import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
v=get_index_daily_data('VIX',days=4000)
vx=v.assign(date=pd.to_datetime(v.date)).drop_duplicates('date').set_index('date').close.reindex(p.index).ffill() if v is not None else pd.Series(index=p.index,dtype=float)
rows=[]
def sig(i):
 if i<125:return None
 r20=p.iloc[i]/p.iloc[i-20]-1; r60=p.iloc[i]/p.iloc[i-60]-1
 vol=r.iloc[i-59:i+1].std().replace(0,np.nan)
 # remove common cross-sectional component, then reward persistent relative trend
 med20=r20.median(); med60=r60.median()
 x=(0.35*(r20-med20)+0.65*(r60-med60))/vol
 # avoid chasing stressed broad selloffs: trend sleeve is active in normal/low stress only
 breadth=(r.iloc[i-4:i+1]>0).mean().mean()
 high=(pd.notna(vx.iloc[i]) and vx.iloc[i]>vx.iloc[max(0,i-60):i].median())
 gate=0.25 if (high or breadth<0.40) else 1.0
 return x*gate
for i,t in enumerate(p.index):
 if i+10>=len(p):continue
 s=sig(i); f=p.shift(-10).iloc[i]/p.iloc[i]-1
 q=pd.DataFrame({'s':s,'f':f}).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(rows,columns=['date','n','ic'])
print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'dates',len(A),'mean_n',round(A.n.mean(),2),'coverage',round(A.n.mean()/15,4))
for name,c in [('full',A.date>=A.date.min()),('online',A.date>=pd.Timestamp('2026-07-16')),('recent',A.date>=A.date.max()-pd.Timedelta(days=370)),('2027+',A.date>=pd.Timestamp('2027-01-01'))]:
 q=A[c].ic; print(name,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [1,5,10,20,40]:
 z=[]
 for i in range(125,len(p)-h):
  s=sig(i); f=p.shift(-h).iloc[i]/p.iloc[i]-1; q=pd.DataFrame({'s':s,'f':f}).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 z=pd.Series(z);print('decay',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
