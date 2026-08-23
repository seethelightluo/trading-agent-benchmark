import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2029-05-30'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
P=pd.DataFrame({s:D[s]['close'] for s in U}).sort_index().loc[:cut]; r=P.pct_change()
# Relative-strength persistence: asset 20d momentum minus contemporaneous cross-asset median,
# gated by agreement of 5d and 20d relative trends; causal and price-only.
csmed=r.rolling(20,min_periods=15).sum().median(axis=1)
m20=r.rolling(20,min_periods=15).sum().sub(csmed,axis=0)
m5=r.rolling(5,min_periods=4).sum().sub(r.rolling(5,min_periods=4).sum().median(axis=1),axis=0)
gate=np.sign(m20)*np.sign(m5)
vol=r.rolling(20,min_periods=15).std()
f=m20.div(vol*np.sqrt(20)).mul(1+0.30*gate)
f.to_csv('scripts/miner_2_20290531_relative_strength_persistence_signal.csv')
def run(h,a=None,b=None):
 x=[];c=[];t=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b):continue
  z=pd.concat([f.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   x.append(spearmanr(z.f,z.y).statistic);c.append(len(z)/15)
   if i:t.append(np.mean(np.sign(f.iloc[i].reindex(z.index))!=np.sign(f.iloc[i-1].reindex(z.index))))
 x=np.asarray(x);return len(x),np.mean(c),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(t)
print('range',P.index.min(),P.index.max(),'assets',len(U),'rows',len(P))
for h in [1,5,10,15,20]:print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-05-30')]:print('regime',a,b,run(10,a,b))
print('artifact','scripts/miner_2_20290531_relative_strength_persistence_signal.csv')
