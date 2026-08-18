import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2031-07-09')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U};p=pd.DataFrame(p).sort_index().loc[:cut];r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]['close'].reindex(p.index).ffill(); z=(vix-vix.rolling(252,min_periods=100).median())/(vix.rolling(252,min_periods=100).std()+1e-8)
res=r.rolling(5,min_periods=4).sum();res=res.sub(res.median(axis=1),axis=0)
# contrarian residual reversal strengthened during elevated volatility regime
f=(-res*(1+z.clip(-1,2)*.35)).shift(1)
def ev(h,idx):
 fw=p.shift(-h)/p-1;x=[];ns=[]
 for d in idx:
  q=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(q)>=8:x.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 x=pd.Series(x).dropna();return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()
print('period',p.index.min().date(),p.index.max().date(),'instruments',15)
for h in [5,10,20]:print('H',h,'dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.4f'%ev(h,f.index))
print('coverage',f.notna().sum().sum()/p.notna().sum().sum(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for n in [365,730,1095]:print('recent',n,'H5 dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.4f'%ev(5,f.index[-n:]))