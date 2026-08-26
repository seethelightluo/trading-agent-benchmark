import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300:d=get_index_daily_data(s,days=6000)
 if d is not None:p[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(p).sort_index(); r=np.log(P).diff()
# Recovery after drawdown: rebound from 60d low, penalized by realized volatility, only when cross-asset breadth is improving.
low=P.rolling(60).min(); recovery=P/low-1
vol=r.rolling(30).std(); breadth=(r>0).mean(axis=1).rolling(10).mean()
# broad improvement gate; lag all inputs by one completed day
f=(recovery/vol).where(breadth.shift(1)>0.50, -recovery/vol).shift(1)
F=P.shift(-10)/P-1
rows=[]; sig=[]
for dt in f.index:
 x=f.loc[dt]; y=F.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].corr(y[ok],method='spearman'),ok.sum()));sig.append((dt,*x.tolist()))
I=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); v=I.ic
print('factor=drawdown_recovery_breadth dates',len(I),'avgN',I.n.mean(),'coverage',I.n.mean()/15)
print('IC10',v.mean(),'ICIR',v.mean()/v.std(),'hit',(v>0).mean(),'recent365',v.tail(365).mean()/v.tail(365).std())
for h in [1,5,10,20]:
 Y=P.shift(-h)/P-1; a=[]
 for dt in f.index:
  x=f.loc[dt];y=Y.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:a.append(x[ok].corr(y[ok],method='spearman'))
 print('decay',h,np.nanmean(a))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
I.to_csv('scripts/miner_2_20341221_drawdown_recovery_ic.csv');pd.DataFrame(sig,columns=['date']+U).set_index('date').to_csv('scripts/miner_2_20341221_drawdown_recovery_signal.csv')
