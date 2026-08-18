import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 if os.path.exists(p):
  q=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'];D[s]=q[q.index<=pd.Timestamp('2034-09-01')]
p=pd.DataFrame(D).sort_index();r=p.pct_change();
# asymmetry: upside/downside realized volatility, low downside relative to upside is desirable
up=r.where(r>0).rolling(60,min_periods=30).std(); dn=r.where(r<0).rolling(60,min_periods=30).std()
sig=(-(dn/(up+1e-12))).shift(1)
for h in [1,5,10,20,40]:
 f=p.shift(-h)/p-1;z=[];ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic);ns.append(len(a))
 z=pd.Series(z).dropna();print(f'h={h} dates={len(z)} avg_n={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={np.mean(z>0):.4f}')
print('artifact coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
sig.to_csv('../persistent/miner_2_20340901_downside_asymmetry_signal.csv')
