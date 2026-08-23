import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2029-03-21'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); dispersion=r.sub(r.mean(axis=1),axis=0).std(axis=1).rolling(20,min_periods=15).mean()
# Dispersion-conditioned short reversal: fades 5d moves, emphasizing cross-asset dispersion regimes,
# with volatility normalization and a 60d relative dispersion z-score.
dz=(dispersion-dispersion.rolling(60,min_periods=40).mean())/dispersion.rolling(60,min_periods=40).std(); mult=(1+0.5*dz.clip(-1,2)).clip(.25,2)
F=(-P.pct_change(5).div(vol*np.sqrt(5),axis=0)).mul(mult,axis=0); F.to_csv('scripts/miner_1_20290322_dispersion_reversal_signal.csv')
def run(h,a=None,b=None):
 x=[];c=[];t=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b):continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   x.append(spearmanr(z.f,z.y).statistic);c.append(len(z)/15)
   if i:t.append(np.mean(np.sign(F.iloc[i].dropna())!=np.sign(F.iloc[i-1].reindex(F.iloc[i].dropna().index))))
 x=np.array(x);return len(x),np.mean(c),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(t)
print('range',P.index.min(),P.index.max(),'assets',len(U))
for h in [5,10,20]:print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-03-21')]:print('regime',a,b,run(10,a,b))
