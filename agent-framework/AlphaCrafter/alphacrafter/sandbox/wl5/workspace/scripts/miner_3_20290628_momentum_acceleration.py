import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2029-06-27'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]; r=P.pct_change()
# Acceleration: recent 20d return relative to prior 20d return, rank cross-section and smooth causally.
m20=r.rolling(20,min_periods=15).sum(); prior20=m20.shift(20); F=(m20-prior20).rank(axis=1,pct=True).rolling(3,min_periods=2).mean()
def run(h,a=None,b=None):
 v=[];c=[];t=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b):continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   v.append(spearmanr(z.f,z.y).statistic);c.append(len(z)/15)
   if i:t.append(np.mean(np.abs(F.iloc[i]-F.iloc[i-1]).dropna()))
 x=np.array(v);return len(x),np.mean(c),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(t)
print('assets',len(U),'dates',len(P),'range',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:print('ALL',h,run(h))
for a,b in [('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-06-27')]:print(a,b,run(5,a,b),run(10,a,b))
