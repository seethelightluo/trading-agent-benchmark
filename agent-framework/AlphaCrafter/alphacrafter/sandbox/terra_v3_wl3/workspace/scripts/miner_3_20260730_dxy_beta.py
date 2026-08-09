import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).query('date<=@cut').set_index('date')['close'].reindex(r.index).ffill().pct_change()
ics={h:[] for h in [1,5,10]}; sig=[]
for i in range(45,len(r)-10):
 x=r.iloc[i-30:i]; y=d.iloc[i-30:i]
 # defensive DXY beta: prefer assets whose beta is low/negative when dollar strengthens
 f=-(x.mul(y,axis=0).mean()-x.mean()*y.mean())/(y.var()+1e-12)
 sig.append((r.index[i],f))
 for h in ics:
  z=pd.concat([f,r.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(z)>=8: ics[h].append((r.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
for h,x in ics.items():
 q=pd.Series(dict(x)); print('H',h,'dates',len(q),'avgN',r.loc[q.index].notna().sum(axis=1).mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=q[(q.index.year>=lo)&(q.index.year<=hi)]; print('regime',lo,hi,'n',len(z),'ic',z.mean())
q=pd.DataFrame({dt:f for dt,f in sig}).T.rank(axis=1,pct=True)
print('coverage',r.loc[q.index].notna().mean().mean(),'turnover',q.diff().abs().mean().mean())
print('cutoff',r.index[-1])
# corr with likely library prototypes
for name,fn in [('mom',lambda i:r.iloc[i-20:i].sum()),('rev',lambda i:-r.iloc[i-5:i].sum()),('clv',lambda i: (pd.read_csv('../persistent/stock_data/SPX.csv') if False else None))]: pass
# pooled rank correlation to momentum and reversal
A=pd.DataFrame({dt:f for dt,f in sig}).T
for w,n in [(20,'mom'),(5,'rev')]:
 B=r.rolling(w).sum().reindex(A.index); print('rho',n,A.stack().corr(B.stack()))
