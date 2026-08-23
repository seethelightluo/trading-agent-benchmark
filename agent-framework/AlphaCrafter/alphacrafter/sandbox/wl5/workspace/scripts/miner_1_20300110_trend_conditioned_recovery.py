import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-01-09')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna();r=p.pct_change();r5=p.pct_change(5);r20=p.pct_change(20);v=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# Recovery factor: buy recent underperformers only when medium trend is positive;
# use cross-sectional relative returns and volatility scaling.
rel5=r5.sub(r5.median(axis=1),axis=0); trend=r20.sub(r20.median(axis=1),axis=0)
s=(-rel5/v)*((trend>0).astype(float)+0.35*(trend<=0).astype(float))
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [5,10,20]:
 z=[];n=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:z.append(q.f.corr(q.y,method='spearman'));n.append(len(q))
 z=pd.Series(z);print('TEST',h,'dates',len(z),'mean_n',round(np.mean(n),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'coverage',round(np.mean(np.array(n)/15),4))
print('turnover',round(s.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'coverage',round(s.notna().mean().mean(),6))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2030-01-09')]:
 z=[]
 for i in range(len(p)-10):
  if pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b):
   q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1:z.append(q.f.corr(q.y,method='spearman'))
 z=pd.Series(z);print('REG10',a,b,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20300110_trend_conditioned_recovery_signal.csv',index=False);print('artifact rows',len(out))
