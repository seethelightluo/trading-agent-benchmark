import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index(); r=np.log(p).diff()
# low recent volatility, lagged
f=(-r.rolling(20,min_periods=15).std()).shift(1)
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vs=[];ds=[]; ns=[]
 for d in p.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):vs.append(q);ds.append(d);ns.append(len(z))
 s=pd.Series(vs,index=ds);print(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(),(s>0).mean())
rank=f.rank(axis=1,pct=True);print('coverage',f.notna().mean().mean(),'turn',rank.diff(10).abs().mean().mean())
