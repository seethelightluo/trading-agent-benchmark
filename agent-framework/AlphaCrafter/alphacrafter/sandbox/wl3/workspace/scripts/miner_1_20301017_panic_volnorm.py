import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None:D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); r3=r.rolling(3).sum(); b=(r3>0).mean(axis=1); v=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# Panic-only volatility-normalized reversal, reducing cross-asset risk bias.
f=-r3.div(v).mul((b<.35).astype(float),axis=0); rows=[]
for t in f.index:
 j=r.index.searchsorted(t,side='right')
 for h in [5,10,20]:
  k=j+h-1
  if j>=len(r) or k>=len(r):continue
  z=pd.concat([f.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
  if len(z)>=8:rows.append((t,h,z.iloc[:,0].corr(z.iloc[:,1])))
x=pd.DataFrame(rows,columns=['date','h','ic']);print('dates',x.date.nunique(),'instruments',len(U),'coverage',round(f.notna().stack().mean(),5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for h in [5,10,20]:
 a=x[x.h==h].ic;print('H',h,'n',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030)]:
  q=x[(x.h==h)&x.date.dt.year.between(lo,hi)].ic;print('REG',lo,hi,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),len(q))
f.to_csv('scripts/miner_1_20301017_panic_volnorm_signal.csv')
