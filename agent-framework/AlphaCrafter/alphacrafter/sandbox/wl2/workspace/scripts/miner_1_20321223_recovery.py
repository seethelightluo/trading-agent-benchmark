import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2032-12-23'); x={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();d=d[d.index<=end]; r=d.close.pct_change(); down=r.where(r<0,0)
 # recovery quality: short return divided by downside risk, emphasizing rebounds with contained downside
 dd=np.sqrt((down**2).rolling(30,min_periods=20).mean())
 x[s]=pd.DataFrame({'f':r.rolling(5).sum()/dd,'c':d.close})
dates=sorted(set().union(*[z.index for z in x.values()])); obs=[]
for dt in dates:
 row=[]
 for s in U:
  z=x[s]
  if dt not in z.index:continue
  i=z.index.get_loc(dt)
  if i<35 or i+10>=len(z):continue
  f=z.iloc[i].f; fw=z.c.iloc[i+10]/z.c.iloc[i]-1
  if np.isfinite(f) and np.isfinite(fw):row.append((f,fw))
 if len(row)>=8:obs.append((dt,spearmanr([a for a,b in row],[b for a,b in row]).statistic,len(row)))
a=np.array([v for d,v,n in obs]);print('idea=downside-adjusted recovery 5d/30d downside vol','dates',len(a),'avg_n',np.mean([n for d,v,n in obs]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for h in [1,3,5,10,20]:
 q=[]
 for dt in dates:
  row=[]
  for s in U:
   z=x[s]
   if dt not in z.index:continue
   i=z.index.get_loc(dt)
   if i<35 or i+h>=len(z):continue
   f=z.iloc[i].f;fw=z.c.iloc[i+h]/z.c.iloc[i]-1
   if np.isfinite(f) and np.isfinite(fw):row.append((f,fw))
  if len(row)>=8:q.append(spearmanr([u for u,v in row],[v for u,v in row]).statistic)
 print('decay',h,np.mean(q))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=[v for d,v,n in obs if lo<=d.strftime('%Y')<=hi];print('regime',lo,hi,len(q),np.mean(q) if q else np.nan,np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
# artifact
rows=[]
for dt,_,_ in obs:
 for s in U:
  if dt in x[s].index and np.isfinite(x[s].loc[dt,'f']):rows.append((dt,s,x[s].loc[dt,'f']))
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20321223_recovery_signal.csv',index=False)
print('artifact_rows',len(rows))
