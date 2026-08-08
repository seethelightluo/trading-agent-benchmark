import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# volatility-of-volatility stability: negative dispersion of 10d realized vol in trailing 60 sessions
rv=r.rolling(10,min_periods=8).std()
f=-rv.rolling(60,min_periods=45).std()
# cross-sectional z/rank values are not needed for Spearman

def calc(h):
 vals=[]; dates=[]; ns=[]
 for i in range(len(p)-h):
  # signal at i, forward return next h observations
  x=f.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1
  ok=x.notna()&y.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(p.index[i]); ns.append(ok.sum())
 s=pd.Series(vals,index=dates)
 return len(s),np.mean(s),np.mean(s)/s.std(ddof=1),np.mean(s>0),np.mean(ns),s
for h in [1,5,10,20]:
 n,ic,ir,hit,mn,s=calc(h); print(f'H{h}: dates={n} meanN={mn:.2f} IC={ic:.6f} ICIR={ir:.6f} hit={hit:.3f}')
# regime H10
n,ic,ir,hit,mn,s=calc(10)
for label,mask in [('2020-23',s.index<'2024-01-01'),('2024-27',(s.index>='2024-01-01')&(s.index<'2028-01-01')),('2028-30',(s.index>='2028-01-01')&(s.index<'2031-01-01')),('2031',s.index>='2031-01-01'),('latest120',s.index>=s.index[-120])]:
 z=s[mask]; print(label,len(z),f'{z.mean():.6f}',f'{z.mean()/z.std(ddof=1):.6f}')
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'turnover10 proxy',np.mean((f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1)).dropna()))
# candidate correlations with basic observable panels / library proxy constructs
for name,q in [('rvmean',-rv.rolling(60,min_periods=45).mean()),('mom20',p.pct_change(20)),('drawdown',p/p.rolling(120,min_periods=60).max()-1)]:
 aa=f.stack(); bb=q.stack(); z=pd.concat([aa,bb],axis=1).dropna(); print('corr',name,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z))
