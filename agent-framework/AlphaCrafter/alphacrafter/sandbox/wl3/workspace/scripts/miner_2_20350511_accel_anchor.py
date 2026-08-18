import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}; p=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index().ffill(); r=p.pct_change(); v=r.rolling(40).std(); q=p.pct_change(20)-p.pct_change(60); q=q/(v*np.sqrt(252)); q=q.sub(q.median(axis=1),axis=0)
eq=U[:8]; de=['XAU','US10Y','CN10Y']; reg=(p[de].pct_change(20).mean(1)-p[eq].pct_change(20).mean(1)).shift(1); x=q.mul(pd.Series(np.where(reg<0,1.,-1.),index=p.index),axis=0)
for n,z in [('cond',x),('plain',q)]:
 a=[]; ns=[]
 for i in range(len(p)-10):
  f=z.iloc[i].dropna(); y=(p.iloc[i+10]/p.iloc[i]-1).reindex(f.index).dropna(); f=f.reindex(y.index)
  if len(f)>=8:a.append(f.corr(y,method='spearman'));ns.append(len(f))
 a=pd.Series(a).dropna(); print(n,len(a),np.mean(ns),a.mean(),a.mean()/a.std(),(a>0).mean(),a.tail(120).mean(),a.tail(252).mean())
x.to_csv('scripts/miner_2_20350511_accel_anchor_signal.csv')
