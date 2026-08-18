import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-05-22'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change()
# downside-efficiency trend: cumulative 20d return normalized by downside volatility, causal
ret=P.pct_change(20); dn=R.where(R<0,0).rolling(20).std(); f=ret/(dn+1e-8); y=P.shift(-10)/P-1

def eval(y):
 a=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(a); return a,np.asarray(ns),np.asarray(ds)
a,n,d=eval(y); print('ALL dates',len(a),'avgN',round(n.mean(),3),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),6))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-05-22')]:
 q=(d>=pd.Timestamp(lo))&(d<=pd.Timestamp(hi)); b=a[q]; print('REG',lo,hi,'dates',len(b),'N',round(n[q].mean(),3),'IC',round(b.mean(),6),'ICIR',round(b.mean()/b.std(ddof=1),6),'hit',round((b>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [1,3,5,10,20]:
 aa,_,_=eval(P.shift(-h)/P-1); print('DECAY',h,'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6))
f.to_csv('scripts/miner_3_20280523_downside_efficiency_signal.csv'); print('period',P.index.min().date(),P.index.max().date())
