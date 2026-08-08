import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];root='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index();r=p.pct_change()
# negative recent volatility change: volatility expansion is expected to mean-revert; cross-sectional signal lagged one day
v10=r.rolling(10,min_periods=8).std()*np.sqrt(10);v40=r.rolling(40,min_periods=25).std()*np.sqrt(40)
f=-(v10/v40-1).shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1;z=[];ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.array(z);print('H',h,'dates',len(z),'N',round(np.mean(ns),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
 for name,lo,hi in [('2020-24','2020','2024-12-31'),('2025-27','2025','2027-12-31'),('2028-29','2028','2029-12-31'),('latest120',str(f.index.max()-pd.Timedelta(days=180)),str(f.index.max()))]:
  sel=(np.array([d>=pd.Timestamp(lo) and d<=pd.Timestamp(hi) for d in f.index]))
  # reconstruct via aligned dates
  dates=[d for d in f.index if d>=pd.Timestamp(lo) and d<=pd.Timestamp(hi)]
  zz=[]
  for d in dates:
   ok=f.loc[d].notna()&y.loc[d].notna()
   if ok.sum()>=8:zz.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic)
  if len(zz):print(' REG',name,len(zz),round(np.mean(zz),5),round(np.mean(zz)/np.std(zz,ddof=1),5))
print('coverage',round(f.notna().stack().mean(),4),'turn10',round((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift(10)).abs().stack().dropna().mean(),4))
