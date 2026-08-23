import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2029-07-25'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std()
shock=(r.rolling(5,min_periods=4).std().mean(axis=1)/r.rolling(60,min_periods=30).std().mean(axis=1)).clip(0.5,2.0)
raw=(-r.rolling(5,min_periods=4).sum()).div(vol.mul(shock,axis=0))
trend=P.pct_change(20); agree=(trend*raw>0).astype(float)
F=(raw*(1-0.35*agree)).rank(axis=1,pct=True)
F.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20290726_volshock_reversal_signal.csv',index=False)
def run(h,a=None,b=None):
 vals=[]; cov=[]; turns=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); cov.append(len(z)/15)
   if i: turns.append(np.mean(np.abs(F.iloc[i]-F.iloc[i-1]).dropna()))
 x=np.array(vals); return len(x),np.mean(cov),float(np.mean(x)),float(np.mean(x)/np.std(x,ddof=1)),float(np.mean(x>0)),float(np.mean(turns))
print('assets',len(U),'rows',len(P),'range',P.index.min().date(),P.index.max().date())
for h in [5,10,20]: print('ALL',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-07-25')]: print(a,b,'10d',run(10,a,b))
