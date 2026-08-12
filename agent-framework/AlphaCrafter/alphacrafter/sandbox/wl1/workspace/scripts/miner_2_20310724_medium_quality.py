import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-07-23')
x={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut] for s in U}; p=pd.DataFrame(x).sort_index().ffill(); r=p.pct_change();
# low-vol quality: medium horizon return, penalized by volatility and max drawdown
f=p.pct_change(60)/(r.rolling(60).std()*np.sqrt(60)+1e-8) - .5*(p/p.rolling(120).max()-1).abs()
for h in [1,5,10,20]:
 q=[]
 for d in p.index:
  z=pd.concat([f.loc[d],(p.shift(-h).loc[d]/p.loc[d]-1)],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna();print(h,len(q),q.mean(),q.mean()/q.std(),(q>0).mean())
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_2_20310724_medium_quality_signal.csv')
