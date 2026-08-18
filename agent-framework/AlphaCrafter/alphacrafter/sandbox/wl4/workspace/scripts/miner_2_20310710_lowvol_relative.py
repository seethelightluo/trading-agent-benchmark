import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-07-09')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U};p=pd.DataFrame(p).sort_index().loc[:cut];r=p.pct_change()
# Relative low-volatility with a mild positive carry overlay: low 30d vol, adjusted by lagged 10d return.
vol=r.rolling(30,min_periods=20).std(); mom=r.rolling(10,min_periods=8).sum(); f=(-vol + .25*mom).shift(1)
def ev(h,idx):
 fw=p.shift(-h)/p-1;x=[];ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=pd.Series(x);return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()
print('period',p.index.min().date(),p.index.max().date(),'instruments',15)
for h in [5,10,20]:print('H',h,'dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.4f'%ev(h,f.index))
print('coverage',f.notna().sum().sum()/p.notna().sum().sum(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for n in [365,730,1095]:print('recent',n,'H10 dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.4f'%ev(10,f.index[-n:]))