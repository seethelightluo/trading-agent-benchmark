import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None: D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Cross-sectional market-neutral medium momentum: each asset's lagged 20d return
# minus the contemporaneous cross-sectional median, scaled by lagged 20d volatility.
# This isolates relative leadership rather than common beta and is available at t-1.
ret20=r.rolling(20).sum(); vol20=r.rolling(20).std()
f=(ret20.shift(1)-ret20.shift(1).median(axis=1),)
f=(ret20.shift(1).sub(ret20.shift(1).median(axis=1),axis=0)).div(vol20.shift(1))
rows=[]
for t in f.index:
 j=r.index.searchsorted(t,side='right')
 for h in [5,10,20]:
  k=j+h-1
  if j>=len(r) or k>=len(r): continue
  z=pd.concat([f.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
  if len(z)>=8: rows.append((t,h,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
x=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('dates',x.date.nunique(),'instruments',len(U),'observations',len(x),'coverage',round(f.notna().stack().mean(),5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for h in [5,10,20]:
 a=x[x.h==h].ic; print('H',h,'n',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030)]:
  q=x[(x.h==h)&x.date.dt.year.between(lo,hi)].ic; print('REG',lo,hi,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),len(q))
f.to_csv('scripts/miner_1_20301128_relative_momentum_signal.csv')
