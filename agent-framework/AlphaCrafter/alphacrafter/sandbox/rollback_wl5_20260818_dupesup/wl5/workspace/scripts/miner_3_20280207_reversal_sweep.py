import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-06'); b=Path('../persistent/stock_data')
px={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}; P=pd.DataFrame(px).sort_index().loc[:end].ffill(); y=P.shift(-10)/P-1
for n in [2,3,5,7,10]:
 r=P.pct_change(n); f=-r.sub(r.median(axis=1),axis=0); a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(a); print(n,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),float(f.notna().mean().mean()),float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
 if n==7:f.to_csv('scripts/miner_3_20280207_relative_reversal_7d_signal.csv')
