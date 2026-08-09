import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data';px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U};P=pd.DataFrame(px).sort_index().loc[:'2026-07-15'];R=P.pct_change();v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.reindex(P.index).ffill();calm=v<v.rolling(252,min_periods=126).median(); mom=R.rolling(20,min_periods=20).sum(); rev=-R.rolling(3,min_periods=3).sum();F=mom.mul(calm,axis=0)+rev.mul(~calm,axis=0)
for h in [1,5,10]:
 vals=[];ns=[];ds=[]
 for i in range(len(F)-h):
  z=pd.concat([F.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(F.index[i])
 a=np.array(vals);print('h',h,'dates',len(a),'avgN',np.mean(ns),'IC %.7f ICIR %.7f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
 if h==1:
  print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(pct=True).diff().abs().mean(axis=1).mean())
  for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-07-15')]:
   q=a[(pd.DatetimeIndex(ds)>=pd.Timestamp(lo))&(pd.DatetimeIndex(ds)<=pd.Timestamp(hi))];print(lab,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('period',P.index.min(),P.index.max(),'assets',P.shape[1]);out=F.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_1_20260730_vix_regime_adaptive_signal.csv');print('signal_artifact scripts/miner_1_20260730_vix_regime_adaptive_signal.csv')
