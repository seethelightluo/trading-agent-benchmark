import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-03-07'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Stress-conditioned reversal: fade recent return, with stronger signal when VIX is elevated
# relative to its trailing 60-session history; normalization avoids raw VIX scale effects.
stress=(vix-vix.rolling(60,min_periods=40).mean())/vix.rolling(60,min_periods=40).std()
stress_mult=(1+0.5*stress.clip(-1,2)).clip(0.25,2.0)
F=(-P.pct_change(5).div(vol*np.sqrt(5),axis=0)).mul(stress_mult,axis=0)
F.to_csv('scripts/miner_3_20290308_vix_conditioned_reversal_signal.csv')
def run(h=10,a=None,b=None):
 vals=[]; cov=[]; turns=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic);cov.append(len(z)/15)
   if i: turns.append(np.mean(np.sign(F.iloc[i].dropna())!=np.sign(F.iloc[i-1].reindex(F.iloc[i].dropna().index))))
 x=np.asarray(vals); return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turns)
print('range',P.index.min(),P.index.max(),'assets',len(U),'vix_dates',vix.notna().sum())
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-03-07')]: print('regime',a,b,run(10,a,b))
print('signal_file','scripts/miner_3_20290308_vix_conditioned_reversal_signal.csv')
