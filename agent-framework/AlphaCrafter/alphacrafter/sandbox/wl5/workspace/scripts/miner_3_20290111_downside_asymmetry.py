import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2029-01-10'].astype(float);r=P.pct_change()
down=r.where(r<0,0.0).rolling(30,min_periods=20).std();tot=r.rolling(30,min_periods=20).std();F=(down/tot).replace([np.inf,-np.inf],np.nan);F.to_csv('scripts/miner_3_20290111_downside_asymmetry_signal.csv')
def run(h=10,a=None,b=None):
 x=[];cov=[];turn=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b):continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:x.append(spearmanr(z.f,z.y).statistic);cov.append(len(z)/15)
  if i:x.append # noop
 x=np.array(x);return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0)
print('range',P.index.min(),P.index.max(),'assets',len(U))
for h in [5,10,20]:print(h,run(h))
for a,b in [('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-01-10')]:print(a,run(10,a,b))
