import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];root='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index();r=p.pct_change()
v10=r.rolling(10,min_periods=8).std()*np.sqrt(10);v40=r.rolling(40,min_periods=25).std()*np.sqrt(40)
# Positive volatility expansion/continuation signal, lagged one completed day.
f=(v10/v40-1).shift(1)
all_ic={}
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; z=[];ns=[]; dates=[]
 for d in f.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic);ns.append(ok.sum());dates.append(d)
 z=np.array(z);all_ic[h]=z
 print('H',h,'dates',len(z),'N',round(np.mean(ns),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
 for name,lo,hi in [('2020-24','2020','2024-12-31'),('2025-27','2025','2027-12-31'),('2028-29','2028','2029-12-31'),('latest120',str(f.index.max()-pd.Timedelta(days=180)),str(f.index.max()))]:
  zz=[v for d,v in zip(dates,z) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)]
  if len(zz)>1:print(' REG',name,len(zz),round(np.mean(zz),5),round(np.mean(zz)/np.std(zz,ddof=1),5))
print('coverage',round(f.notna().stack().mean(),4),'turn10',round((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift(10)).abs().stack().dropna().mean(),4))
