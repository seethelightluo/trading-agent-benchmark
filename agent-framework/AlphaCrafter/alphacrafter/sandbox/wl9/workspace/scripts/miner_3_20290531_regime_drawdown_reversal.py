import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
v=get_index_daily_data('VIX',days=4000)
if v is not None:
 vx=v.assign(date=pd.to_datetime(v.date)).drop_duplicates('date').set_index('date').close.reindex(p.index).ffill()
else: vx=pd.Series(index=p.index,dtype=float)
rows=[]
def signal(i):
 if i<65: return None
 ret20=p.iloc[i]/p.iloc[i-20]-1
 vol=r.iloc[i-19:i+1].std().replace(0,np.nan)
 # cross-asset drawdown/reversal, scaled by idiosyncratic volatility
 dd=p.iloc[i]/p.iloc[i-59:i+1].max()-1
 breadth=(r.iloc[i-4:i+1]>0).mean().mean()
 highv=(vx.iloc[i] > vx.iloc[max(0,i-60):i].median()) if pd.notna(vx.iloc[i]) else False
 # In stressed regimes, buy relative drawdown/reversal; otherwise retain a mild anti-trend signal
 gate=1.0 if (highv or breadth<0.40) else 0.35
 return (-0.65*ret20 - 0.35*dd)/vol*gate
for i,t in enumerate(p.index):
 if i+10>=len(p): continue
 s=signal(i)
 if s is None: continue
 f=p.shift(-10).iloc[i]/p.iloc[i]-1; q=pd.concat([s,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(rows,columns=['date','n','ic'])
print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'dates',len(A),'mean_n',round(A.n.mean(),2),'coverage',round(A.n.mean()/15,4))
for label,cond in [('full',A.date>=A.date.min()),('2026+',A.date>=pd.Timestamp('2026-07-16')),('recent',A.date>=A.date.max()-pd.Timedelta(days=370)),('2027+',A.date>=pd.Timestamp('2027-01-01'))]:
 q=A[cond].ic; print(label,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [1,5,10,20]:
 vals=[]
 for i in range(65,len(p)-h):
  s=signal(i)
  if s is None: continue
  f=p.shift(-h).iloc[i]/p.iloc[i]-1; q=pd.concat([s,f],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 z=pd.Series(vals); print('decay',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
