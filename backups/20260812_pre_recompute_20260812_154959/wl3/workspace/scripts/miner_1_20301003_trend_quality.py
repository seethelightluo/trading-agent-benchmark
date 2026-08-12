import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None:D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Trend-quality: medium-term momentum rewarded when it is persistent, damped by realized risk.
ret=r.rolling(20).sum(); vol=r.rolling(20).std()*np.sqrt(20); consistency=(r>0).rolling(20).mean()
f=(ret/(vol+1e-12))*((0.5+consistency).clip(0.5,1.5))
rows=[]
for t in f.index:
 j=r.index.searchsorted(t,side='right')
 for h in [5,10,20]:
  k=j+h-1
  if j>=len(r) or k>=len(r):continue
  z=pd.concat([f.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
  if len(z)>=8: rows.append((t,h,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
x=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('dates',x.date.nunique(),'instruments',len(U),'coverage',round(f.notna().stack().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for h in [5,10,20]:
 a=x[x.h==h].ic;print('H',h,'obs',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030)]:
  q=x[(x.h==h)&x.date.dt.year.between(lo,hi)].ic
  print('REG',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
f.to_csv('scripts/miner_1_20301003_trend_quality_signal.csv')
