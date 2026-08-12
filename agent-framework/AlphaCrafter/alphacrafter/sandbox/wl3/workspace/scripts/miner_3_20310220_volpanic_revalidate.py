import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:
  try:d=get_index_daily_data(s,4000)
  except Exception:d=None
 if d is not None:P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff(); shock=r.rolling(3,min_periods=3).sum(); rv=r.rolling(20,min_periods=10).std(); breadth=(shock>0).mean(axis=1); csvol=r.std(axis=1).rolling(20,min_periods=10).median(); pct=csvol.rolling(252,min_periods=100).rank(pct=True)
F=(-shock).where((breadth<=.25)&(pct>=.70),0).div(rv)
rows=[]
for t in F.index:
 j=r.index.searchsorted(t,side='right'); k=j+9
 if j>=len(r) or k>=len(r): continue
 z=pd.concat([F.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
 if len(z)>=8: rows.append((t,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']); a=x.ic.dropna(); print('dates',x.date.nunique(),'instruments',len(U),'obs',len(x),'avg_n',round(x.n.mean(),3),'coverage',round(F.notna().stack().mean(),5),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5)); print('10d IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2030),(2031,2031)]:
 q=x[x.date.dt.year.between(lo,hi)].ic; print('REG',lo,hi,round(q.mean(),6) if len(q) else None,round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,len(q))
F.to_csv('scripts/miner_3_20310220_volpanic_revalidated_signal.csv')
