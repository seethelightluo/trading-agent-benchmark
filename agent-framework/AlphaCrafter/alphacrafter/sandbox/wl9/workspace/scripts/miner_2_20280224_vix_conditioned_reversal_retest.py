import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-23')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}).sort_index(); r=px.pct_change(5)
v=pd.read_csv(Path('../persistent/index_data/VIX.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end].reindex(px.index).ffill(); med=v.rolling(60,min_periods=30).median()
# subdued VIX: relative reversal, neutralized cross-section
fac=-(r.sub(r.mean(axis=1),axis=0)).where(v.lt(med),np.nan)
for h in [5,10]:
 fwd=px.shift(-h)/px-1; a=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 a=np.array(a); print(h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)/np.sqrt(len(a))),'hit',(a>0).mean())
 if h==10:
  d=np.array(ds); 
  for lab,m in [('recent252',np.arange(len(a))>=len(a)-252),('2026+',d>=pd.Timestamp('2026-01-01')),('2027+',d>=pd.Timestamp('2027-01-01'))]:
   b=a[m]; print(lab,len(b),b.mean(),b.mean()/(b.std(ddof=1)/np.sqrt(len(b))) if len(b)>1 else np.nan,(b>0).mean())
print('coverage',fac.notna().mean().mean(),'active_dates',fac.notna().any(axis=1).mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2)
