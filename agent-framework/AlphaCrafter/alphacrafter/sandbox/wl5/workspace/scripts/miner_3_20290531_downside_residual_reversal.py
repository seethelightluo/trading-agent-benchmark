import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-05-30'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); neg=r.where(r<0,0.0)
# Residualize each asset's 5d downside shock against contemporaneous cross-sectional mean;
# reverse the idiosyncratic downside component, normalized by downside deviation.
down5=neg.rolling(5,min_periods=5).sum(); cs=down5.mean(axis=1); resid=down5.sub(cs,axis=0)
downvol=np.sqrt((neg**2).rolling(20,min_periods=15).mean())
raw=-resid/(downvol*np.sqrt(5)+1e-12)
# avoid fighting established medium-term trends; retain full reversal only against trend
trend=np.sign(r.rolling(60,min_periods=40).sum())
gated=raw.where(raw.mul(trend)<=0,raw*0.4)
F=gated.rolling(3,min_periods=2).mean().clip(-5,5)
F.to_csv('scripts/miner_3_20290531_downside_residual_reversal_signal.csv')
def run(h=10,a=None,b=None):
 vals=[]; cov=[]; turns=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); cov.append(len(z)/15)
   if i: turns.append(np.mean(np.sign(F.iloc[i].reindex(z.index))!=np.sign(F.iloc[i-1].reindex(z.index))))
 x=np.asarray(vals)
 return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turns)
print('range',P.index.min(),P.index.max(),'assets',len(U),'dates',len(P))
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-05-30')]: print('regime',a,b,run(10,a,b))
print('signal_file','scripts/miner_3_20290531_downside_residual_reversal_signal.csv')
