import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
rev=-(p/p.shift(10)-1)/(r.rolling(60).std()*np.sqrt(10))
trend=p/p.shift(40)-1
f=rev.where(trend>0, 0.0).shift(1)
res={}
for h in [1,5,10,20]:
  y=p.shift(-h)/p-1; ics=[]; ns=[]
  for dt in p.index:
    z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
      ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
  q=pd.Series(ics).dropna(); res[h]={'ic':q.mean(),'icir':q.mean()/q.std(ddof=1),'hit':(q>0).mean(),'dates':len(q),'n':np.mean(ns)}
rr=f.rank(axis=1,pct=True); turn=rr.diff().abs().mean(axis=1).dropna().mean()
print('period',p.index.min(),p.index.max(),'assets',len(p.columns),'rows',len(p))
print(res); print('coverage',f.notna().mean().mean(),'turnover',turn)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20340320_trend_conditioned_reversal_signal.csv',index=False)
