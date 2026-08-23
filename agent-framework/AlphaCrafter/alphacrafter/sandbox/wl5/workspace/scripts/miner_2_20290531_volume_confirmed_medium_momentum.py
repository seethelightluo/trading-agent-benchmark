import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-05-30'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
P=pd.DataFrame({s:D[s]['close'] for s in U}).sort_index().loc[:cut]
V=pd.DataFrame({s:D[s]['volume'] for s in U}).reindex(P.index)
r=P.pct_change()
# Volume-price confirmation: causal medium momentum, strengthened by abnormal turnover
# and signed volume/return agreement. Cross-sectional ranking is left to consumers.
ret=r.rolling(10,min_periods=8).sum()
vol_ratio=V.div(V.rolling(60,min_periods=30).median()).replace([np.inf,-np.inf],np.nan)
signed=(r*vol_ratio).rolling(10,min_periods=8).sum()
confirm=np.tanh(signed/ (r.abs().rolling(20,min_periods=12).mean()*10+1e-12))
f=(ret/(r.rolling(20,min_periods=15).std()*np.sqrt(10))).mul((1+0.35*confirm)*vol_ratio.rolling(5,min_periods=3).mean().clip(.5,2.0))
f.to_csv('scripts/miner_2_20290531_volume_confirmed_medium_momentum_signal.csv')
def run(h,a=None,b=None):
 xs=[];cs=[]; turns=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b): continue
  z=pd.concat([f.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   xs.append(spearmanr(z.f,z.y).statistic); cs.append(len(z)/15)
   if i: turns.append(np.mean(np.sign(f.iloc[i].reindex(z.index))!=np.sign(f.iloc[i-1].reindex(z.index))))
 x=np.asarray(xs); return len(x),np.mean(cs),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turns)
print('range',P.index.min(),P.index.max(),'assets',len(U),'rows',len(P))
for h in [1,5,10,15,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-05-30')]: print('regime',a,b,run(10,a,b))
print('artifact','scripts/miner_2_20290531_volume_confirmed_medium_momentum_signal.csv')
