import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date').close
 px[s]=x
close=pd.DataFrame(px).sort_index().loc[:'2026-07-15']; ret=close.pct_change()
# Low-volatility compression: ratio of recent 5d realized vol to 60d vol, negated.
f=-(ret.rolling(5,min_periods=5).std()/ret.rolling(60,min_periods=40).std()).replace([np.inf,-np.inf],np.nan)
print('period',close.index.min().date(),close.index.max().date(),'universe',len(px))
for h in [1,5,10]:
 fw=close.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r,method='spearman')); ns.append(len(a))
 ic=pd.Series(vals).dropna(); print('h',h,'dates',len(ic),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC %.6f ICIR %.6f hit %.4f std %.6f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),ic.std(ddof=1)))
r=f.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean(),'overall coverage',f.notna().sum().sum()/f.size)
