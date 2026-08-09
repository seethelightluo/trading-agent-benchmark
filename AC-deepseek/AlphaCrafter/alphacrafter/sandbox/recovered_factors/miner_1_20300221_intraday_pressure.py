import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); x=x.set_index('date')
  # signed intraday pressure, normalized by range, smoothed 10d; lagged at observation date
  rng=(x.high-x.low).replace(0,np.nan)
  D[a]=((x.close-x.open)/rng).rolling(10,min_periods=8).mean()
# aligned factor and close returns
F=pd.DataFrame(D); prices=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in D})
rets=prices.pct_change()
for h in [1,5,10,20]:
  vals=[]; dates=[]; ns=[]
  for dt in F.index:
   f=F.loc[dt]; rr=prices.shift(-h).loc[dt]/prices.loc[dt]-1
   z=pd.concat([f,rr],axis=1).dropna();
   if len(z)>=8:
    vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
  q=pd.Series(vals,index=dates).dropna(); ic=q.mean(); sd=q.std(ddof=1); ir=ic/sd*np.sqrt(1) if sd else np.nan
  print('H',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC %.5f ICIR %.5f hit %.3f'%(ic,ir,(q>0).mean()))
  if h==10:
   for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2029-10-01','2030-02-20')]:
    z=q.loc[lo:hi]; print('REG',lo,hi,len(z),round(z.mean(),5),round(z.mean()/z.std(ddof=1),5) if len(z)>1 else np.nan)
# coverage and turnover
print('coverage',F.notna().mean().mean(),'assets',len(F.columns),'dates',len(F))
r=F.rank(axis=1,pct=True); print('turnover10',((r-r.shift(10)).abs().mean(axis=1)).mean())
