import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); px[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(px).sort_index().ffill(); ret=p.pct_change(); bench=ret.mean(axis=1)
rows=[]
def metric(x):
 x=np.asarray(x,float); return len(x),float(np.nanmean(x)),float(np.nanmean(x)/np.nanstd(x,ddof=1)*np.sqrt(252)),float(np.mean(x>0))
# 20d rolling beta-neutral residual return; contrarian residuals predict 10d rebound
for i in range(80,len(p)-10):
 hist=ret.iloc[i-60:i]; b=hist.apply(lambda x: x.cov(bench.iloc[i-60:i])/bench.iloc[i-60:i].var())
 r10=p.iloc[i]/p.iloc[i-10]-1; residual=r10-b*bench.iloc[i-10:i].sum()
 sig=-residual; f=p.iloc[i+10]/p.iloc[i]-1
 q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((p.index[i],q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
A=pd.DataFrame(rows,columns=['date','ic','n'])
print('assets',len(px),'calendar_dates',len(p),'valid_dates',len(A),'mean_n',A.n.mean(),'coverage',A.n.mean()/15)
print('full10d',metric(A.ic))
for name,l,h in [('2020-2023','2020','2024'),('2024-2026','2024','2027'),('2027-2029','2027','2030')]: print(name,metric(A[(A.date>=l)&(A.date<h)].ic))
print('recent252',metric(A.ic.tail(252)))
for h in [5,20]:
 z=[]
 for i in range(80,len(p)-h):
  hist=ret.iloc[i-60:i]; b=hist.apply(lambda x:x.cov(bench.iloc[i-60:i])/bench.iloc[i-60:i].var())
  r10=p.iloc[i]/p.iloc[i-10]-1; sig=-(r10-b*bench.iloc[i-10:i].sum()); f=p.iloc[i+h]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,metric(z))
