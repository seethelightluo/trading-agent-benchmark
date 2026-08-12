import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None:D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=np.log(p).diff()
# Regime-filtered long-horizon trend: lagged 60d relative return, active only
# when the cross-sectional median 20d return is positive; otherwise invert to defensive reversal.
r20=r.rolling(20).sum(); r60=r.rolling(60).sum(); v=r.rolling(20).std()
market=r20.median(axis=1).shift(1)
f=r60.shift(1).sub(r60.shift(1).median(axis=1),axis=0).div(v.shift(1))
f=f.mul(np.where(market.to_numpy()[:,None]>0,1.0,-1.0),axis=0)
rows=[]
for t in f.index:
 j=r.index.searchsorted(t,side='right')
 for h in [5,10,20]:
  if j>=len(r) or j+h-1>=len(r):continue
  z=pd.concat([f.loc[t],r.iloc[j:j+h].sum()],axis=1).dropna()
  if len(z)>=8:rows.append((t,h,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
x=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('dates',x.date.nunique(),'instruments',len(U),'observations',len(x),'coverage',round(f.notna().stack().mean(),5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for h in [5,10,20]:
 a=x[x.h==h].ic;print('H',h,'n',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030)]:
  q=x[(x.h==h)&x.date.dt.year.between(lo,hi)].ic;print('REG',lo,hi,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),len(q))
f.to_csv('scripts/miner_1_20301128_regime_trend_signal.csv')
