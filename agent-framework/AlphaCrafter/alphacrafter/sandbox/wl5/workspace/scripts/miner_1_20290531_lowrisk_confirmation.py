import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-05-30'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); v=r.rolling(20,min_periods=15).std()
# Low-risk factor: inverse trailing volatility, with a causal 5d return
# confirmation term so the signal prefers stable assets that are not in a sharp selloff.
ret5=r.rolling(5,min_periods=5).sum()
F=(1/v)*(1+ret5.clip(-.15,.15))
F=F.replace([np.inf,-np.inf],np.nan)
F.to_csv('scripts/miner_1_20290531_lowrisk_confirmation_signal.csv')
def run(h=10,a=None,b=None):
 vals=[]; cov=[]; turns=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); cov.append(len(z)/15)
   if i: turns.append(np.mean(np.sign(F.iloc[i].reindex(z.index))!=np.sign(F.iloc[i-1].reindex(z.index))))
 x=np.asarray(vals)
 return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turns)
print('range',P.index.min(),P.index.max(),'assets',len(U),'rows',len(P))
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-05-30')]: print('regime',a,b,run(10,a,b))
print('signal_file','scripts/miner_1_20290531_lowrisk_confirmation_signal.csv')
