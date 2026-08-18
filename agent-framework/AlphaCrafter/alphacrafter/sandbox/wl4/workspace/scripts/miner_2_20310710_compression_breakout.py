import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-07-09')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}
p=pd.DataFrame(px).sort_index(); p=p.loc[:cutoff]; r=p.pct_change()
# Breakout continuation: 20d momentum is favored when short volatility has compressed
v5=r.rolling(5,min_periods=5).std(); v30=r.rolling(30,min_periods=20).std(); comp=(v5/(v30+1e-8)).clip(0,5)
mom=r.rolling(20,min_periods=15).sum(); f=(mom/(1+comp)).shift(1)
fr={h:p.shift(-h)/p-1 for h in [5,10,20]}
def ev(fw, idx):
 x=[]; ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=pd.Series(x).dropna();return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()
print('period',p.index.min().date(),p.index.max().date(),'instruments',len(U))
for h,fw in fr.items():print('H',h,'dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.4f'%ev(fw,f.index))
print('coverage',f.notna().sum().sum()/p.notna().sum().sum(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for n in [365,730,1095]:print('recent',n,'H10 dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.4f'%ev(fr[10],f.index[-n:]))