import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; cut=pd.Timestamp('2030-02-20')
x={}
for a in A:
 d=pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); x[a]=d[d.index<=cut]
p=pd.DataFrame(x).sort_index(); r=np.log(p/p.shift(1)); rv=r.rolling(20,min_periods=15).std()
# Contrarian short-term move, scaled by lagged volatility; high score means recent losers with low risk.
f=(-r.rolling(5,min_periods=5).sum()/(rv*np.sqrt(20)+1e-8)).shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; q=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v):q.append(v);ns.append(len(z))
 q=np.array(q); print(h,len(q),round(np.mean(ns),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6),round((q>0).mean(),4),round(q[-250:].mean()/(q[-250:].std(ddof=1))*np.sqrt(252),6))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),6))
