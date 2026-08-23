import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut='2029-04-04'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); v=r.rolling(20,min_periods=15).std(); F=-v
F.to_csv('scripts/miner_1_20290405_lowvol_signal.csv')
def run(h,a=None,b=None):
 x=[];c=[];t=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b):continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   x.append(spearmanr(z.f,z.y).statistic);c.append(len(z)/15)
   if i:t.append(np.mean(np.sign(F.iloc[i].reindex(z.index))!=np.sign(F.iloc[i-1].reindex(z.index))))
 x=np.array(x);return len(x),np.mean(c),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(t)
print('assets',len(U),'dates',len(P))
for h in [5,10,20]:print(h,run(h))
for a,b in [('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-04-04')]:print(a,run(10,a,b))
