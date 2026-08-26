import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); rr=p.pct_change(); bench=rr.mean(axis=1); rows=[]
for i,t in enumerate(p.index):
 if i<65 or i+10>=len(p): continue
 w=p.iloc[i-59:i+1]; rec=p.iloc[i]/w.min()-1; ret20=p.iloc[i]/p.iloc[i-20]-1
 b20=bench.iloc[i-19:i+1].sum(); b60=bench.iloc[i-59:i+1].sum(); stress=(b20<0 and b60<0)
 sig=rec/(0.02+rr.iloc[i-19:i+1].std())
 if stress: sig=sig-0.50*ret20
 for h in [1,5,10]:
  f=p.shift(-h).iloc[i]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna(); q.columns=['s','f']
  if len(q)>=8: rows.append((t,h,len(q),q.s.rank().corr(q.f.rank()),int(stress)))
A=pd.DataFrame(rows,columns=['date','h','n','ic','stress']); print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'rows',len(A))
for h in [1,5,10]:
 q=A[A.h==h]; x=q.ic
 print('H',h,'dates',len(q),'mean_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for label,cond in [('recent252',q.date>=q.date.max()-pd.Timedelta(days=370)),('online',q.date>=pd.Timestamp('2026-07-16')),('ytd',q.date>=pd.Timestamp('2028-01-01'))]:
  y=q[cond].ic; print(label,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else np.nan)
 st=q[q.stress==1]; print('stress_dates',len(st),'stress_ic',round(st.ic.mean(),6) if len(st) else np.nan)
