import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
# Per-asset observation-time rolling skew avoids union-calendar missingness.
F={}; R={}
for s,x in D.items():
 r=np.log(x).diff(); F[s]=r.rolling(60,min_periods=40).skew(); R[s]=x
f=pd.concat(F,axis=1,sort=True).loc[:'2026-07-15']; p=pd.concat(R,axis=1,sort=True).loc[:'2026-07-15']
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h);z=[];ns=[];ds=[]
 for d in f.index:
  a=pd.DataFrame({'f':f.loc[d],'r':fw.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1:z.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));ds.append(d)
 z=np.array(z);ds=pd.DatetimeIndex(ds); sd=z.std(ddof=1)
 print(f'h={h} dates={len(z)} meanN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/sd:.6f} hit={(z>0).mean():.4f}')
 if h==1:
  print(f'coverage={f.notna().sum().sum()/f.size:.6f} turnover={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f}')
  for lab,mask in [('2020-22',ds<='2022-12-31'),('2023-24',(ds>='2023-01-01')&(ds<='2024-12-31')),('2025-26',ds>='2025-01-01')]:
   q=z[mask];print(f'{lab} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f}')
print('period',p.index.min(),p.index.max(),'assets',p.shape[1]);f.to_csv('scripts/miner_1_20260827_return_skew_signal.csv');print('signal_artifact scripts/miner_1_20260827_return_skew_signal.csv')
