import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2029-02-07'].astype(float)
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); m10=P.pct_change(10); m60=P.pct_change(60)
# momentum continuation, with opposing 60d trend damped rather than sign-flipped
F=(m10/vol.replace(0,np.nan))*np.where(m10.mul(m60).ge(0),1.0,0.25)
F.to_csv('scripts/miner_1_20290208_damped_risk_momentum_signal.csv')
def run(h=10,a=None,b=None):
 v=[];c=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b):continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):v.append(q);c.append(len(z)/15)
 x=np.array(v);return len(x),np.mean(c),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.std(x,ddof=1)
print('range',P.index.min().date(),P.index.max().date(),'assets',len(U),'rows',len(P))
for h in [5,10,20]:print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-02-07')]:print('regime',a,b,run(10,a,b))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
