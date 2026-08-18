import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): D[s]=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
c=pd.DataFrame({s:x.close for s,x in D.items()}); r=c.pct_change()
# Recovery-reversal: buy assets with deep 60d drawdowns only when their lagged 10d return has begun recovering.
# Score is contrarian drawdown (negative drawdown) times recovery confirmation, with all inputs lagged one day.
peak=c.rolling(60,min_periods=40).max(); dd=c/peak-1
rec=r.rolling(10,min_periods=8).sum()
vol=r.rolling(20,min_periods=15).std()
f=(-dd.shift(1))*rec.shift(1)/(vol.shift(1)*np.sqrt(252)+1e-12)
print('universe',len(D),'dates',len(c),'assets',c.shape[1])
for h in [5,10,20]:
 fr=c.shift(-h)/c-1; z=[]; ns=[]; dates=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8:
   q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
   if np.isfinite(q): z.append(q);ns.append(len(a));dates.append(d)
 z=pd.Series(z,index=dates); print('horizon',h,'dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
 for n in [365,730,1095]:
  q=z.tail(n); print(' recent',n,'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5),'dates',len(q))
print('coverage',round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
