import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-01-26'); D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=cut].set_index('date')
 r5=d.close.pct_change(5); rv=d.close.pct_change().rolling(20).std()
 vr=d.volume/(d.volume.rolling(20).median().replace(0,np.nan))
 D[s]=pd.DataFrame({'f':r5/(rv*np.sqrt(5)+1e-12)*np.sqrt(vr.clip(0.25,4.0)), 'close':d.close})
all_dates=sorted(set().union(*[set(x.index) for x in D.values()]))
ics=[]; turnovers=[]; valid_counts=[]; prev=None
for dt in all_dates:
 vals=[]
 for s in U:
  x=D[s]
  if dt in x.index:
   loc=x.index.get_loc(dt)
   if loc+1<len(x):
    y=x.close.iloc[loc+1]/x.close.iloc[loc]-1
    if np.isfinite(x.f.loc[dt]) and np.isfinite(y): vals.append((s,x.f.loc[dt],y))
 if len(vals)>=8:
  a=np.array([v[1] for v in vals]); b=np.array([v[2] for v in vals]); ic=spearmanr(a,b).statistic
  if np.isfinite(ic): ics.append((dt,ic)); valid_counts.append(len(vals))
  ranks={v[0]:i for i,v in enumerate(sorted(vals,key=lambda q:q[1]))}
  if prev is not None:
   common=set(prev)&set(ranks)
   if len(common)>=8: turnovers.append(np.mean([abs(ranks[s]/(len(ranks)-1)-prev[s]/(len(prev)-1)) for s in common]))
  prev=ranks
arr=np.array([x[1] for x in ics]); mean=arr.mean(); sd=arr.std(ddof=1)
for h in [1,5,10,20]:
 frames=[]
 for s in U:
  x=D[s]; frames.append(pd.DataFrame({'f':x.f,'y':x.close.shift(-h)/x.close-1}))
 z=pd.concat(frames); vals=[]
 for dt,g in z[z.index<=cut].dropna().groupby(level=0):
  if len(g)>=8: vals.append(spearmanr(g.f,g.y).statistic)
 print('decay',h,round(float(np.nanmean(vals)),6),len(vals))
print('factor=volume_confirmed_risk_momentum_5d dates',len(ics),'universe',15,'avg_n',round(np.mean(valid_counts),3),'coverage',round(len(ics)/len(all_dates),4))
print('IC',round(mean,6),'ICIR',round(mean/sd,6),'hit',round(np.mean(arr>0),4),'std',round(sd,6),'turnover',round(np.mean(turnovers),6),'start',ics[0][0].date(),'end',ics[-1][0].date())
