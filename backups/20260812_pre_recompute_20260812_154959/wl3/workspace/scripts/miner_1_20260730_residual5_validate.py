import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];R=Path('../persistent/stock_data')
def ld(s):return pd.read_csv(R/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'].loc[:'2026-07-15']
p=pd.concat([ld(s).rename(s) for s in S],axis=1).ffill();r=p.pct_change().fillna(0);m=r.mean(1);mu=m.rolling(60).mean(); xc=r-r.rolling(60).mean(); beta=xc.mul(m-mu,axis=0).rolling(60).mean()/((m-mu)**2).rolling(60).mean();f=p.pct_change(5)-beta.mul(m.rolling(5).sum(),axis=0);y=p.pct_change().shift(-1)
def E(y,ix=None):
 q=[];n=[]
 for d in f.index if ix is None else ix:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 q=np.array(q);return len(q),round(np.mean(n),2),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5),round((q>0).mean(),4)
print('daily',E(y));print('5d',E(p.pct_change(5).shift(-5)));print('10d',E(p.pct_change(10).shift(-10)));print('turn',round(f.rank(pct=True).diff().abs().mean().mean(),5))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:print(a,E(y,f.loc[a:b].index))
np.savez('scripts/miner_1_20260730_residual5_signal.npz',dates=f.index.astype(str),symbols=S,values=f.values)
