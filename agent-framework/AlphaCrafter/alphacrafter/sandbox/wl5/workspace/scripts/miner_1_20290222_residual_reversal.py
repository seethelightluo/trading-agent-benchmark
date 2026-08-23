import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end='2029-02-21'
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:end].astype(float)
r=P.pct_change(); m5=P.pct_change(5); vol=r.rolling(20,min_periods=15).std()
# Cross-sectional residual reversal: fade each asset's 5d move relative to the same-day universe median,
# with risk normalization to avoid simply selecting high-volatility instruments.
F=(-(m5.sub(m5.median(axis=1),axis=0))/(vol*np.sqrt(5))).replace([np.inf,-np.inf],np.nan)

def run(h,a=None,b=None):
 vals=[]; ns=[]; dates=[]
 for i in range(len(P)-h):
  d=P.index[i].strftime('%Y-%m-%d')
  if a and not(a<=d<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); dates.append(d)
 x=np.asarray(vals)
 return len(x),float(np.mean(ns)/15),float(x.mean()),float(x.mean()/x.std(ddof=1)),float(np.mean(x>0)),float(pd.DataFrame(F,index=P.index).rank(pct=True).diff().abs().mean(axis=1).mean())
print('range',P.index.min().date(),P.index.max().date(),'assets',len(U),'rows',len(P))
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-02-21')]: print('regime',a,b,run(10,a,b))
print('max factor coverage',F.notna().sum(axis=1).mean()/15)
F.to_csv('scripts/miner_1_20290222_residual_reversal_signal.csv')
