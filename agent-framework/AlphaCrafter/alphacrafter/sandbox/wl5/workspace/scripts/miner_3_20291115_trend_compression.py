import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-11-14'); px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
 px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
# Volatility-compression-confirmed medium trend: 60d return, risk normalized, boosted when recent vol is below its 60d baseline.
mom=p.pct_change(60); compression=(v20/v60).clip(0.5,2.0)
sig=(mom/(v20*np.sqrt(60)).clip(lower=1e-6) * (1.25-0.5*compression)).rank(axis=1,pct=True)
def calc(h,start=None,end=None):
 vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  dt=p.index[i]
  if start and not(pd.Timestamp(start)<=dt<=pd.Timestamp(end)): continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   vals.append(q.f.corr(q.y,method='spearman')); ns.append(len(q)); dates.append(dt)
 x=pd.Series(vals,index=dates).dropna()
 return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0),np.mean(np.array(ns)/15)
print('assets',len(U),'rows',len(p),'range',p.index.min().date(),p.index.max().date())
for h in [5,10,20]: print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-11-14')]: print('REG10',a,b,calc(10,a,b))
print('turnover',sig.diff().abs().mean().mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20291115_trend_compression_signal.csv',index=False); print('artifact_rows',len(out),'latest',out.date.max())
