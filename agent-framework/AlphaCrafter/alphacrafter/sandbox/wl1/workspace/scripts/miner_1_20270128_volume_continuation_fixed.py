import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ds={}
for s in U:
 x=get_stock_daily_data(symbol=s,days=3000)
 if x is not None and len(x)>120:
  x=x.sort_values('date').copy(); x['date']=pd.to_datetime(x.date); ds[s]=x.set_index('date')
calendar=ds['SPX'].index
rows=[]
for d in calendar:
 vals=[]; fw=[]
 for s,x in ds.items():
  if d not in x.index: continue
  k=x.index.get_loc(d)
  if isinstance(k,slice): k=k.stop-1
  if k<65 or k+10>=len(x): continue
  c=pd.to_numeric(x.close,errors='coerce'); v=pd.to_numeric(x.volume,errors='coerce').replace(0,np.nan)
  rv=c.pct_change().iloc[k-19:k+1].std()
  av=np.log(v.iloc[k-4:k+1].mean()/v.iloc[k-59:k+1].mean()) if v.iloc[k-59:k+1].mean()>0 else np.nan
  z=(c.iloc[k]/c.iloc[k-10]-1)*av/max(rv,.003)
  f=c.iloc[k+10]/c.iloc[k]-1
  if np.isfinite(z) and np.isfinite(f): vals.append(z);fw.append(f)
 if len(vals)>=8 and np.std(vals)>0 and np.std(fw)>0: rows.append((d,np.corrcoef(vals,fw)[0,1],len(vals)))
q=pd.DataFrame(rows,columns=['date','ic','n']); print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',np.mean(q.ic>0));print(q.assign(year=q.date.dt.year).groupby('year').ic.mean().to_string());print('decay not run; last',q.tail().to_string(index=False))
