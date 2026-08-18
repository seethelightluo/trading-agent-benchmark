import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2029-04-04')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=cutoff]
prices=pd.DataFrame(px).sort_index().ffill(); ret=prices.pct_change()
r5=prices/prices.shift(5)-1; r20=prices/prices.shift(20)-1; r60=prices/prices.shift(60)-1
agree=(np.sign(r5)+np.sign(r20)+np.sign(r60))/3
vol=ret.rolling(20).std()*np.sqrt(20); f=(r20/(vol+1e-12))*agree

def calc(h):
 rows=[]
 for i in range(60,len(prices)-h):
  x=f.iloc[i]; y=prices.iloc[i+h]/prices.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: rows.append((prices.index[i],ok.sum(),spearmanr(x[ok],y[ok]).statistic))
 return pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
for h in [1,5,10,20]:
 z=calc(h); a=z.ic.values
 print(h,'dates',len(a),'avgN',z.n.mean(),'IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)+1e-12),'hit',(a>0).mean())
z=calc(10); a=z.ic.values
print('coverage',f.loc[:cutoff].notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for n in [250,500]:
 q=a[-n:]; print('recent',n,q.mean(),q.mean()/(q.std(ddof=1)+1e-12))
print('period',z.index.min().date(),z.index.max().date(),'avgN',z.n.mean())
