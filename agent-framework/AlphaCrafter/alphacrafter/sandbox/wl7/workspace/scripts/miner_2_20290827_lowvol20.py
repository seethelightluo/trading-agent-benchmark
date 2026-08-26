import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close
p=pd.DataFrame(p).sort_index().ffill(); r=p.pct_change()
# lagged low-volatility factor: lower trailing 20d realized volatility ranks higher
f=(-r.rolling(20).std()).shift(1); fr=p.pct_change(10).shift(-10)
def run(mask=None):
 a=[]; ns=[]
 for dt in f.index[mask] if mask is not None else f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
print('all',run())
for n,lo,hi in [('2020_22','2020','2022-12-31'),('2023_25','2023','2025-12-31'),('2026_28','2026','2028-12-31'),('2029','2029','2029-08-27')]: print(n,run((f.index>=lo)&(f.index<=hi)))
valid=f.notna().sum(1); print('coverage',valid.mean()/15,'avgN',valid.mean(),'dates',len(f))
f.to_csv('scripts/miner_2_20290827_lowvol20_signal.csv',index_label='date')
