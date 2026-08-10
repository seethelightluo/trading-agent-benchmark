import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
H=10; L=20
xs={}
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
for s in U:
 d=get(s)
 if d is None or len(d)<L+H+2: continue
 d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').drop_duplicates('date')
 d['r']=d.close.pct_change(); d['fac']=(d.close/d.close.shift(L)-1)/(d.r.abs().rolling(L).sum()+1e-12); d['fwd']=d.close.shift(-H)/d.close-1
 xs[s]=d[['date','fac','fwd']].dropna()
all_dates=sorted(set().union(*[set(x.date) for x in xs.values()])); results=[]; turns=[]; prev={}
for dt in all_dates:
 row=[]
 for s,d in xs.items():
  z=d[d.date==dt]
  if len(z): row.append((s,float(z.fac.iloc[0]),float(z.fwd.iloc[0])))
 ranks={s:rank for rank,(s,_) in enumerate(sorted([(s,f) for s,f,_ in row],key=lambda x:x[1]))}
 if prev and len(set(prev)&set(ranks))>=8:
  common=set(prev)&set(ranks); turns.append(np.mean([abs(ranks[s]-prev[s])/(len(common)-1) for s in common]))
 prev=ranks
 if len(row)>=8:
  a=pd.DataFrame(row,columns=['s','f','y']); ic=a.f.corr(a.y,method='spearman')
  if np.isfinite(ic): results.append((dt,ic,len(a)))
ics=np.array([x[1] for x in results]); dates=[x[0] for x in results]; ns=[x[2] for x in results]
print('factor=trend_efficiency L=20 H=10 dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f coverage=%.4f'%(len(ics),np.mean(ns),np.mean(ics),np.mean(ics)/(np.std(ics,ddof=1)+1e-12),np.mean(ics>0),np.mean(turns),len(ics)/(len(all_dates) or 1)))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15'),('2026-07-16','2027-02-25')]:
 q=[v for dt,v in zip(dates,ics) if a<=str(dt)[:10]<=b]; print(a,b,'n',len(q),'IC',np.mean(q) if q else np.nan,'ICIR',np.mean(q)/(np.std(q,ddof=1)+1e-12) if len(q)>1 else np.nan)
out=[]
for s,d in xs.items():
 for _,r in d.iterrows(): out.append({'date':r.date.strftime('%Y-%m-%d'),'symbol':s,'signal':r.fac})
pd.DataFrame(out).to_csv('../persistent/factor_signals_miner_2_20270225_trend_efficiency.csv',index=False)
