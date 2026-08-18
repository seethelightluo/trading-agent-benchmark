import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2026-07-15')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
prices=pd.DataFrame(px).sort_index().loc[:cutoff]; r=prices.pct_change()
risk=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','COPPER','WTI','BTC','ETH']
bench=r[risk].mean(axis=1); down=bench.clip(upper=0).fillna(0)
fac=pd.DataFrame(index=r.index,columns=syms,dtype=float)
for s in syms:
 cov=r[s].rolling(60,min_periods=40).cov(down); var=down.rolling(60,min_periods=40).var(); fac[s]=-cov/var
for h in [1,5,10]:
 fr=prices.pct_change(h).shift(-h); a=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): a.append(q);ds.append(dt);ns.append(len(z))
 a=np.array(a); print('H',h,'Ndates',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean(ns)/15)
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
  b=a[[lo<=str(d)[:4]<=hi for d in ds]]; print(lo,hi,len(b),round(b.mean(),5),round(b.mean()/b.std(ddof=1),5))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'coverage',fac.notna().sum(axis=1).mean()/15)