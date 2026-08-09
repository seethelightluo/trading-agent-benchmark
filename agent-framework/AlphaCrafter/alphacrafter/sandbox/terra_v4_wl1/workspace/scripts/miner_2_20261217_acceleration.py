import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-12-16')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
 P[s]=d[d.index<=cutoff]
p=pd.DataFrame(P).sort_index().ffill(); r=p.pct_change()
# lagged acceleration: recent 5d return less preceding 15d return scaled to same horizon
f=(p.pct_change(5).shift(1) - p.pct_change(15).shift(1)/3)
# smooth with volatility normalization
rv=r.rolling(20).std().shift(1)*np.sqrt(20); f=f/rv
for h in [1,5,10]:
 y=p.pct_change(h).shift(-h); q=[]; ns=[]; ds=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 a=np.array(q); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==1:
  print(pd.Series(a,index=ds).groupby(lambda x:x.year).mean().to_dict())
ranks=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover',round((ranks-ranks.shift()).abs().mean(axis=1).mean(),5),'period',p.index.min().date(),p.index.max().date())
# save artifact for possible admission
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20261217_acceleration_signal.csv',index=False)
