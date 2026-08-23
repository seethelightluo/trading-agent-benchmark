import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2029-06-27'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); med=r.rolling(20,min_periods=15).std().median(axis=1)
# Medium-horizon trend, demeaned cross-section, with causal volatility-state gate: favor trend in calm markets and dampen it in stressed markets.
mom=r.rolling(60,min_periods=40).sum(); z=mom.sub(mom.median(axis=1),axis=0)
stress=med > med.rolling(252,min_periods=100).median()
F=z.where(~stress,z*0.35).rolling(3,min_periods=2).mean()
F.to_csv('scripts/miner_1_20290628_stress_gated_medium_trend_signal.csv')
def run(h,a=None,b=None):
 vals=[]; cov=[]; turns=[]
 for i in range(1,len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b): continue
  q=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   vals.append(spearmanr(q.f,q.y).statistic); cov.append(len(q)/15)
   turns.append(np.mean(np.sign(F.iloc[i])!=np.sign(F.iloc[i-1])))
 x=np.array(vals); return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turns)
print('assets',len(U),'price_dates',len(P),'factor_dates',F.notna().all(axis=1).sum())
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-06-27')]: print('regime',a,b,run(10,a,b))
