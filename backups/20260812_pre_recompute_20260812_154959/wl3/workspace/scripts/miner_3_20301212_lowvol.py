import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None:D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); vol=r.rolling(20).std(); F=-vol
rows=[]
for t in F.index:
 j=r.index.searchsorted(t,side='right')
 for h in [5,10,20]:
  if j+h-1>=len(r):continue
  z=pd.concat([F.loc[t],r.iloc[j:j+h].sum()],axis=1).dropna()
  if len(z)>=8:rows.append((t,h,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
x=pd.DataFrame(rows,columns=['date','h','ic','n']);print('dates',x.date.nunique(),'instruments',len(U),'observations',len(x),'coverage',round(F.notna().stack().mean(),5),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for h in [5,10,20]:
 a=x[x.h==h].ic;print('H',h,'n',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030)]:
  q=x[(x.h==h)&x.date.dt.year.between(lo,hi)].ic;print('REG',lo,hi,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),len(q))
F.to_csv('scripts/miner_3_20301212_lowvol_signal.csv')
