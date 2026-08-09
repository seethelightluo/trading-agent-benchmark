import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# orthogonal medium-term trend: 20d return minus beta-scaled 60d trend, a trend acceleration signal
xs={}
for s in U:
 d=get_stock_daily_data(s,days=2200)
 if d is not None and len(d): xs[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(xs).sort_index(); r=p.pct_change()
# signal available at t, forward return t+1
sig=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
for s in p:
 rr=r[s]
 sig[s]=rr.rolling(20,min_periods=20).sum()-rr.rolling(60,min_periods=60).sum()/3.0
rows=[]
for i in range(len(p)-1):
 a=sig.iloc[i]; f=r.iloc[i+1]
 z=pd.concat([a,f],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  rows.append((p.index[i],ic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/len(U))
print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for q in [0,1,2,3]:
 sub=x.iloc[q*len(x)//4:(q+1)*len(x)//4];print('regime',q+1,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
# save exact signal artifact
out=sig.copy();out.index.name='date';out.to_csv('../persistent/factor_signals_miner_2_20270225_trend_acceleration.csv')
