import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data/'
cs={}
for s in U:
 d=pd.read_csv(base+s+'.csv',parse_dates=['date']).set_index('date').close
 cs[s]=d
px=pd.DataFrame(cs).sort_index().loc[:'2026-07-15']
ret=px.pct_change(5)
rows=[]
for dt in ret.index:
 x=ret.loc[dt]
 if x.notna().sum()<8: continue
 # leave-one-out cross-asset lagged impulse, available at dt close; forward starts next day
 f=x.copy()
 for s in U: f[s]=x.drop(labels=s).median()
 # cross-sectional signal is common-ish but leave-one-out creates rank differences
 for h in [1,5,10]:
  fut=px.shift(-h).loc[dt]/px.loc[dt]-1
  z=pd.concat([f,fut],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   rows.append((dt,h,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=r[r.h==h].ic
 print('h',h,'dates',len(q),'meanIC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'median',q.median(),'mean_n',r[r.h==h].n.mean())
print('coverage',len(ret.dropna(how='all')),'instruments',len(U))
# regime halves
for h in [1,5,10]:
 q=r[r.h==h].set_index('date').ic
 for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15')]:
  v=q.loc[a:b]; print(h,a,'n',len(v),'ic',v.mean(),'icir',v.mean()/v.std(ddof=1) if len(v)>1 else np.nan)
