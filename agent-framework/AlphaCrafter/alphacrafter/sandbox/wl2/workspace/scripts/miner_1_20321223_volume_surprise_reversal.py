import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2032-12-23')
xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 d=d[d.index<=end]; r=d.close.pct_change()
 # high-volume selloff reversal: recent return, scaled by abnormal volume; lag naturally via signal at t predicting t+10
 vol=np.log1p(d.volume.replace(0,np.nan))
 vz=(vol-vol.rolling(60,min_periods=30).mean())/vol.rolling(60,min_periods=30).std()
 f=-(r.rolling(5).sum())*vz.clip(-3,3)
 xs[s]=pd.DataFrame({'f':f,'close':d.close})
all_dates=sorted(set().union(*[x.index for x in xs.values()]))
obs=[]; vals=[]
for dt in all_dates:
 row=[]
 for s in U:
  x=xs[s]
  if dt not in x.index: continue
  i=x.index.get_loc(dt)
  if i<65 or i+10>=len(x): continue
  # only if forward interval exists and dates before cutoff
  f=x.iloc[i].f; fw=x.close.iloc[i+10]/x.close.iloc[i]-1
  if np.isfinite(f) and np.isfinite(fw): row.append((s,f,fw))
 if len(row)>=8:
  a=np.array([z[1] for z in row]); b=np.array([z[2] for z in row])
  ic=spearmanr(a,b).statistic
  obs.append((dt,ic,len(row))); vals.extend(row)
ics=np.array([z[1] for z in obs]);
print('idea=volume-surprise selloff reversal; dates',len(obs),'avg_n',np.mean([z[2] for z in obs]),'universe',len(U))
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0),'coverage',len(obs)/len(all_dates))
for h in [1,3,5,10,20]:
 oo=[]
 for dt in all_dates:
  row=[]
  for s in U:
   x=xs[s]
   if dt not in x.index: continue
   i=x.index.get_loc(dt)
   if i<65 or i+h>=len(x): continue
   if i+10>=len(x): continue
   f=x.iloc[i].f; fw=x.close.iloc[i+h]/x.close.iloc[i]-1
   if np.isfinite(f) and np.isfinite(fw): row.append((f,fw))
  if len(row)>=8: oo.append(spearmanr([a for a,b in row],[b for a,b in row]).statistic)
 print('decay',h,np.nanmean(oo),len(oo))
# rank turnover
ranks=[]
for dt,_,_ in obs:
 a=[]
 for s in U:
  x=xs[s]
  if dt in x.index: a.append((s,x.loc[dt,'f']))
 ranks.append(dict((s,i) for i,(s,_) in enumerate(sorted(a,key=lambda z:z[1]))))
t=[]
for a,b in zip(ranks,ranks[1:]):
 common=set(a)&set(b)
 if len(common)>=8:t.append(np.mean([a[s]!=b[s] for s in common]))
print('turnover_rank_change',np.mean(t))
# regime
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 z=[v for d,v,n in obs if lo<=d.strftime('%Y')<=hi]
 print('regime',lo,hi,'dates',len(z),'ic',np.mean(z) if z else np.nan,'icir',np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
# artifact
out=pd.DataFrame([(d,s,xs[s].loc[d,'f']) for d,_,_ in obs for s in U if d in xs[s].index and np.isfinite(xs[s].loc[d,'f'])],columns=['date','symbol','signal'])
out.to_csv('scripts/miner_1_20321223_volume_surprise_reversal_signal.csv',index=False)
print('artifact_rows',len(out))
