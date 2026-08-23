import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-04-18'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Cross-sectional residual short reversal: remove each day's median 3d move,
# scale by idiosyncratic 20d volatility, then damp signals aligned with 20d trend.
ret3=r.rolling(3).sum(); common=ret3.median(axis=1)
resid=ret3.sub(common,axis=0)
raw=-resid.div(vol*np.sqrt(3))
trend=np.sign(r.rolling(20,min_periods=15).sum())
F=raw.where(raw.mul(trend)<=0,raw*0.5)
F.to_csv('scripts/miner_3_20290419_cs_residual_reversal_signal.csv')
def run(h=10,a=None,b=None):
 vals=[]; cov=[]; turns=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); cov.append(len(z)/15)
   if i>0: turns.append(np.mean(np.sign(F.iloc[i].reindex(z.index))!=np.sign(F.iloc[i-1].reindex(z.index))))
 x=np.asarray(vals)
 return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turns)
print('range',P.index.min(),P.index.max(),'assets',len(U),'valid_rows',F.dropna(how='all').shape[0])
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-04-18')]: print('regime',a,b,run(10,a,b))
print('signal_file','scripts/miner_3_20290419_cs_residual_reversal_signal.csv')
