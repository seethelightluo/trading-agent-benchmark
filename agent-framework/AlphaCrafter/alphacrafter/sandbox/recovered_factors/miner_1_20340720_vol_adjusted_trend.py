import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 xs[a]=d['close']
p=pd.concat(xs,axis=1).sort_index()
r=np.log(p).diff()
# candidate: volatility-adjusted medium trend, lagged one session
ret20=np.log(p/p.shift(20)); vol20=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(ret20/vol20).shift(1)
# require broad enough values; cross-sectional IC by date
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h)
 vals=[]; ns=[]; dates=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
 s=pd.Series(vals,index=dates).dropna()
 print('H',h,'dates',len(s),'meanN',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean()))
 if h==1:
  print('regimes',[(y,len(s[str(y)]),s[str(y)].mean(),s[str(y)].mean()/s[str(y)].std()) for y in [2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034]])
# turnover rank every 10 sessions
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'meanN',f.notna().sum(1).mean(),'turn10',rank.diff(10).abs().mean().mean())
