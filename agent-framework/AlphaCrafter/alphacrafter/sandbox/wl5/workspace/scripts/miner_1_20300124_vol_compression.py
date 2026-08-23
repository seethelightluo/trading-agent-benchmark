import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-01-23'); px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date'])
 px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change()
# Volatility compression: recent 5d realized volatility relative to trailing 60d volatility.
# Lower compression ratio is hypothesized to precede positive 10d returns (quiet assets recover).
v5=r.rolling(5,min_periods=5).std(); v60=r.rolling(60,min_periods=40).std()
s=(-(v5/v60)).replace([np.inf,-np.inf],np.nan)
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [3,5,10,20]:
 xs=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: xs.append(q.f.corr(q.y,method='spearman')); ns.append(len(q))
 x=pd.Series(xs); print('TEST',h,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4))
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',s.notna().mean().mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2030-01-23')]:
 xs=[]
 for i in range(len(p)-10):
  if pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b):
   q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8: xs.append(q.f.corr(q.y,method='spearman'))
 x=pd.Series(xs);print('REG10',a,b,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20300124_vol_compression_signal.csv',index=False)
print('artifact rows',len(out))
