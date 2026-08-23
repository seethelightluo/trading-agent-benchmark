import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2029-10-31'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); mom=r.rolling(60,min_periods=40).sum()
# Defensive persistence: prefer low realized risk, but only when medium trend is positive.
F=(-vol.rank(axis=1,pct=True)+0.4*mom.rank(axis=1,pct=True)).rolling(3,min_periods=2).mean()
def run(h,a=None,b=None):
 vals=[]; cov=[]; turns=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); cov.append(len(z)/15)
   if i>0: turns.append(np.mean(F.iloc[i].rank(pct=True).values != F.iloc[i-1].rank(pct=True).values))
 vals=np.array(vals); return len(vals),float(np.mean(cov)),float(np.mean(vals)),float(np.mean(vals)/np.std(vals,ddof=1)),float(np.mean(vals>0)),float(np.mean(turns))
print('assets',len(U),'dates',len(P),'cutoff',cut)
for h in [3,5,10,20]: print('H',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-10-31')]: print('REG',a,b,run(5,a,b))
F.to_csv('scripts/miner_1_20291101_lowvol_trend_signal.csv')
