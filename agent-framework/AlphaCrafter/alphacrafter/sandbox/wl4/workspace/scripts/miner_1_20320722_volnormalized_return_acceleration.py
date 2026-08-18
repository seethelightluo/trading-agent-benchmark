import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): D[s]=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
c=pd.DataFrame({s:x.close for s,x in D.items()}); r=c.pct_change()
# Acceleration: recent 10-session return minus one-third of prior 30-session return,
# scaled by trailing 60-session volatility; lagged one session.
recent=r.rolling(10,min_periods=8).sum()
prior=r.rolling(30,min_periods=20).sum().shift(10)
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
f=((recent-prior/3)/(vol+1e-12)).shift(1)
print('universe',len(D),'dates',len(c),'assets',c.shape[1])
for h in [5,10,20]:
 fr=c.shift(-h)/c-1; z=[]; ns=[]; ds=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8:
   q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
   if np.isfinite(q): z.append(q);ns.append(len(a));ds.append(d)
 z=pd.Series(z,index=ds); print('horizon',h,'dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
 for n in [365,730,1095]:
  q=z.tail(n); print('recent',n,'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5),'dates',len(q))
 print('regimes',[(a,b,round(z.loc[(z.index>=a)&(z.index<=b)].mean(),5),round(z.loc[(z.index>=a)&(z.index<=b)].mean()/z.loc[(z.index>=a)&(z.index<=b)].std(ddof=1),5)) for a,b in [('2020','2027-12-31'),('2028','2030-12-31'),('2031','2032-12-31')]])
print('coverage',round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
