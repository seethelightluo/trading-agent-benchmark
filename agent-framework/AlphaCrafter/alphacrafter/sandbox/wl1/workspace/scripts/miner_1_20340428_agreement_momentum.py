import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<150:d=get_index_daily_data(s,4000)
 if d is not None and len(d): fs[s]=pd.Series(d.close.values,index=pd.to_datetime(d.date),name=s)
p=pd.concat(fs.values(),axis=1).sort_index(); r=np.log(p).diff();
# Agreement-weighted intermediate momentum: direction persistence times 20/60 residual momentum, lagged
bench=r.mean(axis=1); res=r.sub(bench,axis=0)
a=(res.rolling(20).sum()/ (res.rolling(60).std()*np.sqrt(60)+1e-10)); agree=(np.sign(res).rolling(40).mean()).abs(); f=(a*agree).shift(1)
for h in [5,10,20]:
 z=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1]))
 print('h',h,'dates',len(z),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z),'hit',np.mean(np.array(z)>0))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().sum(axis=1).mean()/15)
f.to_csv('scripts/miner_1_20340428_agreement_momentum_signal.csv')
