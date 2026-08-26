import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}; vols={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  z=d[['date','close','volume']].copy(); z.date=pd.to_datetime(z.date); z=z.drop_duplicates('date').set_index('date')
  prices[s]=z.close; vols[s]=z.volume
p=pd.DataFrame(prices).sort_index().ffill(); v=pd.DataFrame(vols).reindex(p.index).ffill()
rows=[]
# Volume-confirmed cross-sectional reversal: recent losers are favored, but only
# when their current volume is unusually high relative to their own 20d baseline.
for i,t in enumerate(p.index):
 if i<45 or i+10>=len(p): continue
 r5=p.iloc[i]/p.iloc[i-5]-1
 resid=r5-r5.median()
 vr=(v.iloc[i]/v.iloc[i-20:i].mean()).replace([np.inf,-np.inf],np.nan)
 sig=-resid*(0.5+0.5*vr.clip(0,3))
 f=p.iloc[i+10]/p.iloc[i]-1
 q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
A=pd.DataFrame(rows,columns=['date','ic','n'])
def met(z):
 z=np.asarray(z,dtype=float); return (len(z),float(np.nanmean(z)),float(np.nanmean(z)/np.nanstd(z,ddof=1)*np.sqrt(252)) if len(z)>1 else np.nan,float(np.mean(z>0)))
print('assets',len(prices),'calendar_dates',len(p),'valid_dates',len(A),'mean_n',A.n.mean(),'coverage',A.n.mean()/15)
print('10d',met(A.ic))
for name,l,h in [('2020-2023','2020','2024'),('2024-2026','2024','2027'),('2027-2029','2027','2030'),('recent252',None,None)]:
 z=A.ic.tail(252) if name=='recent252' else A[(A.date>=l)&(A.date<h)].ic
 print(name,met(z))
# Same signal evaluated at alternate horizons for decay
for h in [5,20]:
 z=[]
 for i in range(45,len(p)-h):
  r5=p.iloc[i]/p.iloc[i-5]-1; resid=r5-r5.median(); vr=(v.iloc[i]/v.iloc[i-20:i].mean()).replace([np.inf,-np.inf],np.nan)
  sig=-resid*(.5+.5*vr.clip(0,3)); f=p.iloc[i+h]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,met(z))
