import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
Ds={};
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date'); Ds[s]=d
p=pd.concat({s:d.close for s,d in Ds.items()},axis=1,sort=True).loc[:'2026-07-15']
# gap/open-to-prev-close, lagged 3-day mean; reverse gap as signal
f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s,d in Ds.items():
 q=(d.open/d.close.shift(1)-1).dropna(); a=-q.rolling(3,min_periods=3).mean(); f.loc[a.index,s]=a
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[];ns=[];dates=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));dates.append(dt)
 z=np.array(vals); d=pd.DatetimeIndex(dates); sd=z.std(ddof=1)
 print(f'h={h} dates={len(z)} meanN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/sd:.6f} hit={(z>0).mean():.4f}')
 if h==1:
  print(f'coverage={f.notna().sum().sum()/f.size:.6f} turnover={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f}')
  for lab,mask in [('2020-22',d<='2022-12-31'),('2023-24',(d>='2023-01-01')&(d<='2024-12-31')),('2025-26',d>='2025-01-01')]:
   q=z[mask];print(f'{lab} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f}')
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_3_20260827_gap_signal.csv');print('period',p.index.min(),p.index.max(),'assets',p.shape[1]);print('signal_artifact scripts/miner_3_20260827_gap_signal.csv')
