import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  z=d[['date','open','close','high','low']].copy(); z.date=pd.to_datetime(z.date)
  z=z.drop_duplicates('date').set_index('date'); frames[s]=z
# Factor: 5-day return reversal weighted by recent rejection (close location / range).
# Negative 5d return is attractive; repeated closes near the opposite end of daily ranges indicate exhaustion.
clv={}; ret={}; close={}
for s,z in frames.items():
 rg=(z.high-z.low).replace(0,np.nan)
 clv[s]=((2*z.close-z.high-z.low)/rg).rolling(5).mean()
 ret[s]=z.close.pct_change(5); close[s]=z.close
C=pd.DataFrame(close).sort_index().ffill(); R=pd.DataFrame(ret).reindex(C.index); L=pd.DataFrame(clv).reindex(C.index)
rows=[]
for i,t in enumerate(C.index):
 if i<10 or i+10>=len(C): continue
 # Reversal is stronger when 5d loss coincides with persistent close-location rejection.
 sig= -R.iloc[i] * (-L.iloc[i])
 for h in [1,5,10]:
  f=C.iloc[i+h]/C.iloc[i]-1
  q=pd.concat([sig,f],axis=1).dropna(); q.columns=['s','f']
  if len(q)>=8 and q.s.nunique()>1: rows.append((t,h,len(q),q.s.rank().corr(q.f.rank())))
A=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('range',C.index.min().date(),C.index.max().date(),'assets',len(frames),'rows',len(A))
for h in [1,5,10]:
 q=A[A.h==h]; x=q.ic
 print('H',h,'dates',len(q),'mean_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for label,cond in [('recent252',q.date>=q.date.max()-pd.Timedelta(days=370)),('online',q.date>=pd.Timestamp('2026-07-16')),('ytd',q.date>=pd.Timestamp('2028-01-01'))]:
  y=q[cond].ic; print(label,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else np.nan)
