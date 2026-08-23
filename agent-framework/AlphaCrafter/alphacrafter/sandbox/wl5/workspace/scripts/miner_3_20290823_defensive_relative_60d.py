import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-08-22'); p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']); p[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(p).sort_index().dropna(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); basket=p[['XAU','US10Y','CN10Y']].pct_change(60).mean(axis=1)
rel=p.pct_change(60).sub(basket,axis=0); disp=vol.mean(axis=1); gate=(disp>disp.rolling(120,min_periods=60).median()).astype(float)
base=-rel/(vol.clip(lower=1e-5)*np.sqrt(252)); sig=base.mul(1+.35*gate,axis=0).rank(axis=1,pct=True)
def calc(h,a,b):
 z=[];ns=[];ds=[]
 for i in range(len(p)-h):
  dt=p.index[i]
  if not(pd.Timestamp(a)<=dt<=pd.Timestamp(b)):continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(q.f.corr(q.y,method='spearman'));ns.append(len(q));ds.append(dt)
 x=pd.Series(z,index=ds).dropna();return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)
print('rows',len(p))
for h in [5,10,20]:print('ALL',h,calc(h,'2020-01-01','2029-08-22'))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-08-22')]:print('REG',a,b,calc(10,a,b))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20290823_defensive_relative_60d_signal.csv',index=False);print('artifact',len(out))
