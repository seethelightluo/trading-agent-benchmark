import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None:D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); r3=r.rolling(3).sum()
# Volatility-scaled relative shock reversal, conditioned on elevated cross-sectional dispersion.
med=r3.median(axis=1); resid=r3.sub(med,axis=0)
disp=resid.abs().median(axis=1); threshold=disp.rolling(120,min_periods=60).median()
vol=r.rolling(20,min_periods=10).std()*np.sqrt(252)
# cap scaling to avoid domination by unusually quiet instruments
scale=(1.0/vol).clip(upper=1/vol.median(axis=1).replace(0,np.nan),axis=0)
f=-resid.mul(scale,axis=0).mul((disp>threshold).astype(float),axis=0)
rows=[]
for t in f.index:
 j=r.index.searchsorted(t,side='right')
 for h in [1,3,5,10,20]:
  k=j+h-1
  if j>=len(r) or k>=len(r):continue
  z=pd.concat([f.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
  if len(z)>=8:rows.append((t,h,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
x=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('range',p.index.min(),p.index.max(),'dates',x.date.nunique(),'instruments',len(U),'observations',len(x),'coverage',round(f.notna().stack().mean(),5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for h in [1,3,5,10,20]:
 a=x[x.h==h].ic; print('H',h,'n',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030)]:
  q=x[(x.h==h)&x.date.dt.year.between(lo,hi)].ic
  if len(q): print('REG',lo,hi,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),len(q))
for lo in ['2028-01-01','2029-01-01','2030-01-01']:
 q=x[(x.h==5)&(x.date>=lo)].ic
 if len(q):print('RECENT',lo,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),len(q))
f.to_csv('scripts/miner_2_20301128_volscaled_relative_shock_signal.csv')
