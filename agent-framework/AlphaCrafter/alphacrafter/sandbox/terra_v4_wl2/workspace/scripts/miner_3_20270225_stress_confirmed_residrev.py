import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); r=P.pct_change(); vol=r.rolling(20).std(); ret3=P.pct_change(3); resid=ret3.sub(ret3.median(axis=1),axis=0)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill(); disp=r.rolling(3).std().mean(axis=1); va=vol.mean(axis=1)/vol.mean(axis=1).rolling(60).median()
# stress confirmation: high dispersion, high VIX, and rising volatility; residual reversal
active=(disp>disp.rolling(60).median())&(vix>vix.rolling(120).median())&(va>1.0)
f=(-resid/vol).where(active); artifact='../persistent/factor_signals_miner_3_20270225_stress_confirmed_residrev.csv'; f.to_csv(artifact)
def calc(y):
 a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 return np.array(a)
q=calc(P.pct_change(1).shift(-1)); print('dates',len(q),'active',active.sum(),'avg_n',f.notna().sum(axis=1)[f.notna().sum(axis=1)>=8].mean(),'coverage',f.notna().sum().sum()/f.size); print('h1 IC %.9f ICIR %.9f hit %.5f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for h in [3,5]:
 q=calc(P.pct_change(h).shift(-h)); print('h',h,'IC',q.mean(),'n',len(q))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027-02-25')]:
 # date filter via index reconstruction
 z=[]
 for dt in f.loc[a:b].index:
  x=pd.concat([f.loc[dt],P.pct_change().shift(-1).loc[dt]],axis=1).dropna()
  if len(x)>=8:z.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
 print(a,b,len(z),np.mean(z) if z else np.nan)
