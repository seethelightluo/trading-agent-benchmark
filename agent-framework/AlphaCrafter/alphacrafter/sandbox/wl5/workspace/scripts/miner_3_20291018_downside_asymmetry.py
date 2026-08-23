import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2029-10-17');px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna();r=p.pct_change();v=r.rolling(20,min_periods=15).std();loss=r.rolling(5).sum();dd=p/p.rolling(60,min_periods=40).max()-1
# downside-asymmetry reversal: only materially negative shocks, scaled by drawdown and volatility.
s=(-loss/v.clip(lower=1e-5)).where((loss/v.clip(lower=1e-5)<-0.75)&(dd<0),0.)
s=s.rank(axis=1,pct=True)
for h in [5,10,20]:
 x=[];n=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:x.append(q.f.corr(q.y,method='spearman'));n.append(len(q));ds.append(p.index[i])
 x=pd.Series(x,index=ds);print('ALL',h,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',np.mean(np.array(n)/15))
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20291018_downside_asymmetry_signal.csv',index=False)
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean(),'artifact_rows',len(out))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-10-17')]:
 z=[]
 for i in range(len(p)-10):
  if pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b):
   q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1:z.append(q.f.corr(q.y,method='spearman'))
 z=pd.Series(z);print('REG10',a,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
