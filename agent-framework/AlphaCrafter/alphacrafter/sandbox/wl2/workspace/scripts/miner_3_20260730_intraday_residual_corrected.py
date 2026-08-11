import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
I=pd.DataFrame({s:-(x.close/x.open-1) for s,x in D.items()}).sort_index()
C=pd.DataFrame({s:-(2*(x.close-x.low)/(x.high-x.low).replace(0,np.nan)-1) for s,x in D.items()}).sort_index()
Y=pd.DataFrame({s:x.close.shift(-1)/x.close-1 for s,x in D.items()}).sort_index()
vals=[]; raw=[]; ns=[]
for dt in I.index:
 z=pd.DataFrame({'i':I.loc[dt],'c':C.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.c.values]; b=np.linalg.lstsq(X,z.i.values,rcond=None)[0]; f=z.i.values-X@b
  vals.append(spearmanr(f,z.y).statistic); raw.append(spearmanr(z.i,z.y).statistic); ns.append(len(z))
a=np.array(vals); print('dates',len(a),'meanN',np.mean(ns),'coverage',len(a)/len(I),'raw IC/ICIR',np.mean(raw),np.mean(raw)/np.std(raw,ddof=1),'res IC/ICIR',a.mean(),a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for h in [5,10]:
 Yh=pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U}).sort_index(); q=[]
 for dt in I.index:
  z=pd.DataFrame({'i':I.loc[dt],'c':C.loc[dt],'y':Yh.loc[dt]}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.c.values]; b=np.linalg.lstsq(X,z.i.values,rcond=None)[0]; q.append(spearmanr(z.i.values-X@b,z.y).statistic)
 q=np.array(q); print(h,len(q),q.mean(),q.mean()/q.std(ddof=1))
for y in [2020,2021,2022,2023,2024,2025,2026]:
 q=[]
 for dt in I.loc[str(y)].index:
  z=pd.DataFrame({'i':I.loc[dt],'c':C.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.c.values]; b=np.linalg.lstsq(X,z.i.values,rcond=None)[0]; q.append(spearmanr(z.i.values-X@b,z.y).statistic)
 print(y,len(q),round(np.mean(q),5) if q else None)
print('turnover',I.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'coverage_valid',I.notna().sum().sum()/I.size)
