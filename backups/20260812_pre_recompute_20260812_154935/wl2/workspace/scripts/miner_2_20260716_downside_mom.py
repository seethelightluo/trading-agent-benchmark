import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
close=pd.DataFrame(px).sort_index().loc[:'2026-07-15']; r=close.pct_change()
# Downside-risk-adjusted medium momentum: 40d return divided by trailing 60d downside deviation.
down=r.where(r<0).rolling(60,min_periods=30).std()
f=(close.pct_change(40)/down).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10]:
 fw=close.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a))
 q=pd.Series(vals); print(h,len(q),np.mean(ns),np.mean(ns)/15,q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0),q.std(ddof=1))
print('turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().sum().sum()/f.size)
