import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U},axis=1,sort=True).loc[:'2026-10-21']
r=np.log(P).diff(); f=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 q=r[s].dropna(); neg2=q.clip(upper=0).pow(2)
 dd=np.sqrt(neg2.rolling(30,min_periods=20).mean())
 sig=q.rolling(30,min_periods=20).sum()/dd.replace(0,np.nan)
 f.loc[sig.index,s]=sig
f.to_csv('scripts/miner_1_20261022_downside_adjusted_momentum_signal.csv')
for h in [1,5,10]:
 fw=np.log(P).shift(-h)-np.log(P); z=[];ns=[];ds=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1:
   z.append(spearmanr(a.f,a.r).statistic); ns.append(len(a)); ds.append(dt)
 s=pd.Series(z,index=pd.DatetimeIndex(ds)); print(f'h={h} dates={len(s)} avgN={np.mean(ns):.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={(s>0).mean():.4f}')
 if h==1:
  print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
  for label,mask in [('2020-22',s.index<='2022-12-31'),('2023-24',(s.index>='2023-01-01')&(s.index<='2024-12-31')),('2025-26',s.index>='2025-01-01')]:
   q=s[mask];print(label,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('period',P.index.min(),P.index.max(),'assets',P.shape[1])
