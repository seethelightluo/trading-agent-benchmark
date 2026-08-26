import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); d=d.set_index('date'); px[s]=d.close; vol[s]=d.volume
p=pd.DataFrame(px).sort_index().ffill(); v=pd.DataFrame(vol).reindex(p.index).ffill(); r=p.pct_change()
# Volume-confirmed reversal: fade recent 5d move, amplified when volume is unusually high.
volsur=v.rolling(20).mean().div(v.rolling(60).mean()).replace([np.inf,-np.inf],np.nan)
f=(-(r.rolling(5).sum())*volsur).shift(1); fr=p.pct_change(10).shift(-10)
def calc(idx):
 a=[]; ns=[]
 for dt in f.index[idx]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
print('all',calc(np.ones(len(f),bool)))
for n,lo,hi in [('2020_22','2020','2022-12-31'),('2023_25','2023','2025-12-31'),('2026_28','2026','2028-12-31'),('2029','2029','2029-09-09')]: print(n,calc((f.index>=lo)&(f.index<=hi)))
print('coverage',f.notna().sum(1).mean()/15,'avgN',f.notna().sum(1).mean(),'dates',len(f))
f.to_csv('scripts/miner_2_20290910_volume_confirmed_reversal_signal.csv',index_label='date')
