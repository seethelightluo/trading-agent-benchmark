import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-06-22')
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
for s in U:px[s]=px[s][px[s].index<=end]
dates=sorted(set.intersection(*[set(v.index) for v in px.values()]))
def calc(dt,h):
 vals={s:x.loc[:dt] for s,x in px.items()}; r20={s:(z.iloc[-1]/z.iloc[-21]-1) if len(z)>=22 else np.nan for s,z in vals.items()}; med=np.nanmedian(list(r20.values())); f={s:-(r20[s]-med) for s in U if np.isfinite(r20[s])}; y={}
 for s,z in vals.items():
  fut=px[s][px[s].index>dt].head(h)
  if len(fut)==h and s in f:y[s]=fut.iloc[-1]/z.iloc[-1]-1
 c=set(f)&set(y)
 return (spearmanr([f[s] for s in c],[y[s] for s in c]).statistic,len(c)) if len(c)>=8 else (np.nan,0)
rows_by={}
for h in [1,5,10,20]:
 a=[];ns=[];rows=[]
 for dt in dates:
  q,n=calc(dt,h)
  if n>=8:a.append(q);ns.append(n);rows.append((dt,q,n))
 print('horizon',h,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(np.array(a)>0))
 rows_by[h]=rows
r=pd.DataFrame(rows_by[10],columns=['date','ic','n']).set_index('date');
for label,lo,hi in [('pre','2020-01-01','2029-12-31'),('post','2030-01-01','2033-06-22'),('recent',end-pd.Timedelta(days=365),end)]:
 q=r[(r.index>=pd.Timestamp(lo))&(r.index<=pd.Timestamp(hi))].ic;print(label,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
r.reset_index().to_csv('scripts/miner_2_20330623_relative_strength_spread_ic.csv')
# signal artifact on latest date
sig=[];dt=dates[-1]; z={s:px[s].loc[:dt] for s in U}; rr={s:z[s].iloc[-1]/z[s].iloc[-21]-1 for s in U}; m=np.median(list(rr.values()))
for s in U:sig.append((dt,s,-(rr[s]-m)))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20330623_relative_strength_spread_signal.csv',index=False)
print('latest',dt)
