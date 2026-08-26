import pandas as pd,numpy as np
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2034-06-07'); frames={}
for s in symbols:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=end].set_index('date');r=d.close.pct_change()
 # downside-quality reversal: reverse 20d return, scaled by downside deviation, damped by loss frequency
 dd=r.where(r<0,0).rolling(40,min_periods=30).std(); loss=(-r.where(r<0,0)).rolling(20,min_periods=15).mean()
 frames[s]=pd.DataFrame({'f':-r.rolling(20,min_periods=15).sum()/(dd*np.sqrt(20)+1e-12)*(1+loss.rolling(10,min_periods=8).mean()),'close':d.close})
all_dates=sorted(set.intersection(*[set(x.index) for x in frames.values()])); rows=[]; sig=[]
for dt in all_dates:
 va=[];fb=[]; ss=[]
 for s in symbols:
  x=frames[s];p=x.index.get_loc(dt)
  if p+10>=len(x):continue
  a=x.iloc[p].f;b=x.iloc[p+10].close/x.iloc[p].close-1
  if np.isfinite(a) and np.isfinite(b):va.append(a);fb.append(b);ss.append((s,a))
 if len(va)>=8:
  q=spearmanr(va,fb).statistic
  if np.isfinite(q):rows.append((dt,q,len(va)));sig += [(dt,s,a) for s,a in ss]
ser=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]);print('dates',len(ser),'meanN',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15);print('IC',ser.mean(),'dailyICIR',ser.mean()/ser.std(ddof=1),'hit',(ser>0).mean())
for a,b in [('2026','2027'),('2028','2029'),('2030','2032'),('2033','2034')]:
 z=ser[(ser.index.year>=int(a))&(ser.index.year<=int(b))];print(a,b,len(z),z.mean())
for h in [5,20]:
 q=[]
 for dt in all_dates:
  va=[];fb=[]
  for s in symbols:
   x=frames[s];p=x.index.get_loc(dt)
   if p+h>=len(x):continue
   if np.isfinite(x.iloc[p].f):va.append(x.iloc[p].f);fb.append(x.iloc[p+h].close/x.iloc[p].close-1)
  if len(va)>=8:q.append(spearmanr(va,fb).statistic)
 print('h',h,'IC',np.mean(q),'dailyICIR',np.mean(q)/np.std(q,ddof=1))
out='scripts/miner_2_20340608_downside_quality_reversal_signal.csv';pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv(out,index=False);print('artifact',out)
