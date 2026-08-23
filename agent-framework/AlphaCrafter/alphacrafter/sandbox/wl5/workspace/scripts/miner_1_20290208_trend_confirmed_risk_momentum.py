import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2029-02-07'].astype(float)
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); mom10=P.pct_change(10); mom60=P.pct_change(60)
# risk-adjusted short momentum, only retain momentum aligned with the slower 60d trend; neutralize daily cross-sectional level
raw=mom10/vol.replace(0,np.nan)
gate=np.where(mom60.ge(0),1.0,-1.0)
F=(raw*gate).replace([np.inf,-np.inf],np.nan)
F.to_csv('scripts/miner_1_20290208_trend_confirmed_risk_momentum_signal.csv')
def run(h=10,a=None,b=None):
 vals=[]; cov=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append(q); cov.append(len(z)/15)
 x=np.asarray(vals); return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.std(x,ddof=1)
print('range',P.index.min().date(),P.index.max().date(),'assets',len(U),'rows',len(P))
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-02-07'),('2029-01-01','2029-02-07')]: print('regime',a,b,run(10,a,b))
# rank turnover
ranks=F.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)).dropna().mean(); print('turnover',turn)
