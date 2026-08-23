import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-10-03'); px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
 px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
# Acceleration: recent 20d return minus its 60d-average daily pace, risk normalized; continuation hypothesis.
acc=p.pct_change(20)-p.pct_change(60)/3
sig=(acc/(v*np.sqrt(20)).clip(lower=1e-6)).rank(axis=1,pct=True)
def calc(h,start=None,end=None):
 vals=[]; ns=[]; dates=[]; tr=[]
 for i in range(60,len(p)-h):
  dt=p.index[i]
  if start and not(pd.Timestamp(start)<=dt<=pd.Timestamp(end)): continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   vals.append(q.f.corr(q.y,method='spearman')); ns.append(len(q)); dates.append(dt)
   if i: tr.append(sig.iloc[i].sub(sig.iloc[i-1]).abs().mean())
 x=pd.Series(vals,index=dates).dropna(); return len(x),float(np.mean(ns)),float(x.mean()),float(x.mean()/x.std(ddof=1)),float(np.mean(x>0)),float(np.mean(np.array(ns)/15)),float(np.mean(tr))
print('assets',len(U),'rows',len(p),'range',p.index.min().date(),p.index.max().date())
for h in [5,10,20]: print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-10-03')]: print('REG10',a,b,calc(10,a,b))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20291004_trend_acceleration_signal.csv',index=False); print('artifact_rows',len(out),'latest',out.date.max())
