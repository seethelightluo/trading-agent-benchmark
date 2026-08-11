import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames=[]
for s in U:
 d=get_stock_daily_data(symbol=s,days=310)
 if d is not None:
  q=d.sort_values('date')[['date','close']].copy(); q['ret']=q.close.pct_change(); q=q.rename(columns={'close':s,'ret':s+'_r'}); frames.append(q)
# Inner join means explicit common-date cross section
p=frames[0]
for q in frames[1:]: p=p.merge(q,on='date',how='inner')
p=p.sort_values('date').reset_index(drop=True); dates=p.date
ics={5:[],10:[],20:[]}; turns=[]; cov=[]; prev=None
for j in range(65,len(p)-20):
 vals={}; fut={h:{} for h in ics}
 for s in U:
  r=p[s+'_r'].to_numpy(float); px=p[s].to_numpy(float)
  sig=(px[j]/px[j-20]-1)/max(np.std(r[j-19:j+1]),.002)
  sig*=np.clip(np.std(r[j-19:j-1])/max(np.std(r[j-9:j+1]),.002),.5,1.5)
  vals[s]=sig
  for h in ics: fut[h][s]=px[j+h]/px[j]-1
 for h,a in ics.items():
  x=np.array([vals[s] for s in U]); y=np.array([fut[h][s] for s in U])
  if np.std(x)>1e-12 and np.std(y)>1e-12: a.append(pd.Series(x).corr(pd.Series(y),method='spearman'))
 ranks=pd.Series(vals).rank(pct=True); turns.append(np.mean(np.abs(ranks-(prev if prev is not None else ranks)))); prev=ranks; cov.append(1.0)
print('dates',len(p),'usable',len(cov),'avgN',len(U),'coverage',np.mean(cov),'turnover',np.mean(turns))
for h,a in ics.items():
 a=np.asarray(a); print(h,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'n',len(a))
 for label,start in [('2026+', '2026-01-01'),('2027+','2027-01-01'),('2028YTD','2028-01-01')]:
  # date mask approximate based on valid signals date
  ds=dates.iloc[65:65+len(a)].to_numpy(); z=a[ds>=np.datetime64(start)]
  print(' ',label,round(np.nanmean(z),6),round(np.nanmean(z)/np.nanstd(z,ddof=1),6),len(z))
rows=[]
for j in range(65,len(p)-20):
 row={'date':str(dates.iloc[j])}
 for s in U:
  r=p[s+'_r'].to_numpy(float); px=p[s].to_numpy(float)
  row[s]=(px[j]/px[j-20]-1)/max(np.std(r[j-19:j+1]),.002)*np.clip(np.std(r[j-19:j-1])/max(np.std(r[j-9:j+1]),.002),.5,1.5)
 rows.append(row)
pd.DataFrame(rows).to_csv('scripts/miner_1_20280629_stable_voltrend_signal.csv',index=False)
