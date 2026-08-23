import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2029-04-18'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
D=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
r=P.pct_change(); d=D.pct_change()
# Residual short-horizon reversal after removing each asset's rolling 60d DXY exposure.
beta=r.rolling(60,min_periods=40).cov(d).div(d.rolling(60,min_periods=40).var(),axis=0)
res=r.sub(beta.mul(d,axis=0))
shock=res.rolling(5,min_periods=4).sum()
F=-shock
F.to_csv('scripts/miner_1_20290419_dxy_residual_reversal_signal.csv')
def run(h,a=None,b=None):
 vals=[];cov=[];turn=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b):continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic);cov.append(len(z)/15)
   if i: turn.append(np.mean(np.sign(F.iloc[i].reindex(z.index))!=np.sign(F.iloc[i-1].reindex(z.index))))
 x=np.array(vals); return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turn)
print('range',P.index.min(),P.index.max(),'assets',len(U),'rows',len(P))
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-04-18')]: print('regime',a,b,run(10,a,b))
