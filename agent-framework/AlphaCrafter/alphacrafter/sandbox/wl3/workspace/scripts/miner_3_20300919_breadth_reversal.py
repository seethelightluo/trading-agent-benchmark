import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None:D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=np.log(p).diff()
# market breadth conditioned short-horizon reversal: fade 3d moves only in broad selloffs/buyoffs
r3=r.rolling(3).sum(); breadth=(r3>0).mean(axis=1)
f=-r3.mul(((breadth<.35)|(breadth>.65)).astype(float),axis=0)
rows=[]
for t in f.index:
 j=r.index.searchsorted(t,side='right')
 for h in [5,10,20]:
  k=j+h-1
  if j>=len(r) or k>=len(r):continue
  z=pd.concat([f.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
  if len(z)>=8:rows.append((t,h,z.iloc[:,0].corr(z.iloc[:,1])))
x=pd.DataFrame(rows,columns=['date','h','ic']);print('dates',x.date.nunique(),'instruments',len(U),'coverage',f.notna().stack().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [5,10,20]:
 a=x[x.h==h].ic;print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030)]:
  q=x[(x.h==h)&(x.date.dt.year.between(lo,hi))].ic;print(lo,round(q.mean(),5),round(q.mean()/q.std(ddof=1),4),len(q))
f.to_csv('scripts/miner_3_20300919_breadth_reversal_signal.csv')
