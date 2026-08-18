import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cut=pd.Timestamp('2030-02-20')
x={}
for a in A:
 d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 x[a]=d[d.index<=cut]
p=pd.DataFrame(x).sort_index(); lr=np.log(p/p.shift(1))
# lagged signal: medium trend return normalized by recent realized volatility
f=(np.log(p/p.shift(60))/lr.rolling(20).std()).shift(1)
print('candidate=60d_log_return_over_20d_vol; cutoff',cut.date())
for h in [1,5,10,20]:
 r=p.shift(-h)/p-1; q=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],r.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 q=np.asarray(q); print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),6),'hit',round((q>0).mean(),4),'minN',min(ns));
 if len(q)>=250:
  y=q[-250:]; print('recent250_ICIR',round(y.mean()/y.std(ddof=1)*np.sqrt(252),6))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),6))
