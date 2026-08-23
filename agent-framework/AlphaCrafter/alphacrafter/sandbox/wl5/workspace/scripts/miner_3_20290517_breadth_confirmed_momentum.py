import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-05-16'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); v=r.rolling(20,min_periods=15).std()
mom=r.rolling(20,min_periods=15).sum().div(v*np.sqrt(20))
breadth=(r.rolling(20,min_periods=15).sum()>0).mean(axis=1)
# retain trend only when cross-sectional breadth confirms direction; otherwise damp it
F=mom.copy()
F[breadth<0.40]=F[breadth<0.40]*0.35
F[breadth>0.60]=F[breadth>0.60]*1.15
F=F.clip(-4,4)
F.to_csv('scripts/miner_3_20290517_breadth_confirmed_momentum_signal.csv')
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
 return len(x),float(np.mean(cov)),float(np.mean(x)),float(np.mean(x)/np.std(x,ddof=1)),float(np.mean(x>0)),float(np.mean(turns))
print('range',P.index.min(),P.index.max(),'assets',len(U),'rows',len(P))
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-05-16')]: print('regime',a,b,run(10,a,b))
print('signal_file','scripts/miner_3_20290517_breadth_confirmed_momentum_signal.csv')
