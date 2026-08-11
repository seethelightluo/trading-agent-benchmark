import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
r=pd.DataFrame({a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()['close'].pct_change() for a in assets}).sort_index()
# One idea: volatility-scaled short-term reversal, evaluated at multiple forward horizons.
vol=r.rolling(20,min_periods=15).std()
raw=r.rolling(5,min_periods=4).sum()
factor=-(raw/vol.replace(0,np.nan))
for h in [5,10,20]:
  ic=[]; ns=[]; dates=[]
  for i in range(20,len(r)-h-1):
    f=factor.iloc[i]; fr=r.iloc[i+1:i+1+h].sum(min_count=h-2)
    z=pd.concat([f,fr],axis=1).dropna()
    if len(z)>=8:
      ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(r.index[i])
  x=np.asarray(ic); print(f'h={h} dates={len(x)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={np.mean(x>0):.4f}')
  for s in ['2025-01-01','2026-01-01','2027-01-01']:
    q=x[np.asarray(dates)>=pd.Timestamp(s)]
    print(' ',s,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
factor.to_csv('scripts/miner_1_20280127_volscaled_reversal_signal.csv',index_label='date')
