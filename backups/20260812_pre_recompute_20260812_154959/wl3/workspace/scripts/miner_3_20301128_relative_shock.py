import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None: D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); r3=r.rolling(3).sum(); vol=r.rolling(20).std()
br=(r3>0).mean(axis=1)
# Relative shock reversal: in broad selloffs, fade each asset's 3d return relative to cross-sectional median,
# scaled by its own trailing volatility. All state variables use data through signal date t.
med=r3.median(axis=1); shock=(r3.sub(med,axis=0)).div(vol.replace(0,np.nan)); active=(br<=.25)
F=(-shock*active.astype(float).values[:,None]).replace([np.inf,-np.inf],np.nan)
rows=[]
for t in F.index:
 j=r.index.searchsorted(t,side='right')
 for h in [5,10,20]:
  k=j+h-1
  if j>=len(r) or k>=len(r): continue
  z=pd.concat([F.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
  if len(z)>=8: rows.append((t,h,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
x=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('dates',x.date.nunique(),'instruments',len(U),'observations',len(x),'coverage',round(F.notna().stack().mean(),5),'active',round((F!=0).mean().mean(),5),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for h in [5,10,20]:
 a=x[x.h==h].ic; print('H',h,'n',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030)]:
  q=x[(x.h==h)&x.date.dt.year.between(lo,hi)].ic; print('REG',lo,hi,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),len(q))
F.to_csv('scripts/miner_3_20301128_relative_shock_signal.csv')
