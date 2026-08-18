import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2030-01-23')
x={}
for a in A:
 d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'].sort_index();x[a]=d[d.index<=cut]
p=pd.DataFrame(x).sort_index(); lr=np.log(p/p.shift(1)); f=(-lr.rolling(20).std()).shift(1)
for h in [1,5,10,20]:
 r=p.shift(-h)/p-1; q=[];n=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],r.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 q=np.array(q);print(h,len(q),round(np.mean(n),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6),round((q>0).mean(),4),min(n));y=q[-250:];print('recent',round(y.mean(),6),round(y.mean()/y.std(ddof=1)*np.sqrt(252),6))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean())
