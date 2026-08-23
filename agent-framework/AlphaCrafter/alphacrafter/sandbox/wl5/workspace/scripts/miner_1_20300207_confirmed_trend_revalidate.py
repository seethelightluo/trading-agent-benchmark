import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-02-06')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']); px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change(); r10=p.pct_change(10); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
rel=r10.sub(r10.median(axis=1),axis=0); breadth=(r10>0).mean(axis=1); confirm=(2*breadth-1).clip(-1,1); s=(rel/vol).mul(confirm,axis=0)
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [3,5,10,20]:
 x=[]; n=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:x.append(q.f.corr(q.y,method='spearman'));n.append(len(q))
 z=pd.Series(x); print('TEST',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'coverage',round(np.mean(np.array(n)/15),4))
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',s.notna().mean().mean())
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20300207_confirmed_trend_signal.csv',index=False);print('artifact rows',len(out))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2029-01-01'),('2029-01-02','2030-02-06')]:
 x=[]
 for i in range(len(p)-5):
  if not (pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b)): continue
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+5]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:x.append(q.f.corr(q.y,method='spearman'))
 z=pd.Series(x);print('REGIME',a,b,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
