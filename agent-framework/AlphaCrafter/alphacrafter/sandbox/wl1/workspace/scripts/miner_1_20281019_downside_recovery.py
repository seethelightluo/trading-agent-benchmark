import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames=[]
for s in U:
 d=get_stock_daily_data(symbol=s,days=3500)
 if d is not None and len(d)>0:
  q=d.sort_values('date')[['date','close']].rename(columns={'close':s}); frames.append(q)
p=frames[0]
for q in frames[1:]: p=p.merge(q,on='date',how='inner')
p=p.sort_values('date').reset_index(drop=True); dates=pd.to_datetime(p.date)
ics={5:[],10:[],20:[]}; ds={h:[] for h in ics}; turns=[]; prev=None; rows=[]
for j in range(65,len(p)-20):
 vals={}; fut={h:{} for h in ics}
 for s in U:
  px=p[s].to_numpy(float); rr=pd.Series(px).pct_change().to_numpy()
  down=np.std(rr[max(1,j-39):j+1][rr[max(1,j-39):j+1]<0]) if np.any(rr[max(1,j-39):j+1]<0) else .002
  sig=(px[j]/px[j-20]-1)/max(down*np.sqrt(40),.002)
  vals[s]=sig
  for h in ics:fut[h][s]=px[j+h]/px[j]-1
 for h in ics:
  x=pd.Series([vals[s] for s in U]); y=pd.Series([fut[h][s] for s in U]); c=x.corr(y,method='spearman')
  if np.isfinite(c):ics[h].append(c);ds[h].append(dates.iloc[j])
 ranks=pd.Series(vals).rank(pct=True);turns.append(np.mean(np.abs(ranks-(prev if prev is not None else ranks))));prev=ranks
 rows.append({'date':str(dates.iloc[j]),**vals})
print('shared_dates',len(p),'usable_dates',len(rows),'avgN',len(U),'coverage',1.0,'turnover',round(np.mean(turns),6))
for h,a in ics.items():
 a=np.array(a);print(h,'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'n',len(a))
 for label,start in [('2026+','2026-01-01'),('2027+','2027-01-01'),('2028+','2028-01-01')]:
  z=a[np.array(ds[h])>=pd.Timestamp(start)]
  print(label,len(z),round(z.mean(),6) if len(z) else 'nan',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else 'nan')
pd.DataFrame(rows).to_csv('scripts/miner_1_20281019_downside_recovery_signal.csv',index=False)
