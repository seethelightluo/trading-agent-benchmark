import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2029-01-24'].astype(float); r=P.pct_change()
# reversal strength, attenuated for high realized risk
vol=r.rolling(20,min_periods=15).std(); F=(-P.pct_change(5)/(vol*np.sqrt(5))).replace([np.inf,-np.inf],np.nan)
F.to_csv('scripts/miner_3_20290125_lowvol_reversal_signal.csv')
def run(h=10,a=None,b=None):
 v=[];c=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b):continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:v.append(spearmanr(z.f,z.y).statistic);c.append(len(z)/15)
 x=np.asarray(v);return len(x),np.mean(c),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0)
print('range',P.index.min(),P.index.max(),'assets',len(U))
for h in [5,10,20]:print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-01-24')]:print('regime',a,b,run(10,a,b))
