import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2029-06-13'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]; r=P.pct_change()
vol=r.rolling(20,min_periods=15).std(); mom=r.rolling(60,min_periods=40).sum(); F=(-vol.rank(axis=1,pct=True)+0.4*mom.rank(axis=1,pct=True)).rolling(3,min_periods=2).mean()
F.to_csv('scripts/miner_3_20290614_lowvol_trend_signal.csv')
def run(h,a=None,b=None):
 x=[];c=[];t=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b):continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:x+=[spearmanr(z.f,z.y).statistic];c+=[len(z)/15];t+=[np.mean(np.sign(F.iloc[i])!=np.sign(F.iloc[i-1]))] if i else []
 x=np.array(x);return len(x),np.mean(c),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(t)
print('assets',len(U),'dates',len(P))
for h in [5,10,20]:print(h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-06-13')]:print(a,run(5,a,b))
