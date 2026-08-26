import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2033-06-08')
px={}
for s in UNIV:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=end]
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]))
# normalized trend acceleration: recent 10d return minus preceding 20d return, divided by 20d vol
records=[]
for dt in dates:
 vals={}; fw={}
 for s,x in px.items():
  if dt not in x.index: continue
  z=x.loc[:dt]
  if len(z)<45: continue
  r10=z.iloc[-1]/z.iloc[-11]-1
  rprev=z.iloc[-11]/z.iloc[-31]-1
  vol=z.pct_change().iloc[-21:].std()*np.sqrt(252)
  if vol>0 and len(x.loc[dt:])>10:
   vals[s]=(r10-rprev/2)/vol
   future=x[x.index>dt].head(10)
   if len(future)==10: fw[s]=future.iloc[-1]/x.loc[dt]-1
 common=set(vals)&set(fw)
 if len(common)>=8:
  a=np.array([vals[s] for s in common]); b=np.array([fw[s] for s in common])
  records.append((dt,spearmanr(a,b).statistic,len(common)))
r=pd.DataFrame(records,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avgN',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
for n in [1,5,10,20]:
 out=[]
 for dt in dates:
  vals={}; fw={}
  for s,x in px.items():
   z=x.loc[:dt]
   if len(z)<45 or len(x[x.index>dt])<n: continue
   r10=z.iloc[-1]/z.iloc[-11]-1; rp=z.iloc[-11]/z.iloc[-31]-1; v=z.pct_change().iloc[-21:].std()*np.sqrt(252)
   if v>0:
    vals[s]=(r10-rp/2)/v; f=x[x.index>dt].head(n)
    if len(f)==n: fw[s]=f.iloc[-1]/x.loc[dt]-1
  c=set(vals)&set(fw)
  if len(c)>=8: out.append(spearmanr([vals[s] for s in c],[fw[s] for s in c]).statistic)
 print('horizon',n,'IC',np.nanmean(out),'obs',len(out))
# periods
for label,lo,hi in [('pre2030','2020-01-01','2029-12-31'),('2030plus','2030-01-01','2033-06-08'),('recent365',end-pd.Timedelta(days=365),end)]:
 q=r[(r.index>=pd.Timestamp(lo))&(r.index<=pd.Timestamp(hi))].ic
 print(label,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# signal artifact
rows=[]
for dt in r.index:
 for s,x in px.items():
  z=x.loc[:dt]
  if len(z)>=45:
   v=z.pct_change().iloc[-21:].std()*np.sqrt(252)
   if v>0: rows.append({'date':dt,'symbol':s,'signal':(z.iloc[-1]/z.iloc[-11]-1-(z.iloc[-11]/z.iloc[-31]-1)/2)/v})
pd.DataFrame(rows).to_csv('scripts/miner_2_20330609_trend_acceleration_signal.csv',index=False)
r.reset_index().to_csv('scripts/miner_2_20330609_trend_acceleration_ic.csv',index=False)
