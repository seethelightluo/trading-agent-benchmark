import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; m=get_index_daily_data('DXY',days=3000); P={s:get_stock_daily_data(s,days=3000) for s in U}; px=pd.DataFrame({s:d.set_index('date').close for s,d in P.items()}).sort_index(); dx=m.set_index('date').close.sort_index(); r=np.log(px).diff(); mr=np.log(dx).diff().reindex(r.index)
vm=mr.rolling(60,min_periods=45).var(); f=pd.DataFrame(index=r.index)
for s in U:
 x=r[s]; cov=(x*mr).rolling(60,min_periods=45).mean()-x.rolling(60,min_periods=45).mean()*mr.rolling(60,min_periods=45).mean(); f[s]=-cov/vm
print('assets',len(U))
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1]))
 q=pd.Series(q);print(h,len(q),round(q.mean(),5),round(q.mean()/q.std(),5),round((q>0).mean(),4))
print('coverage',f.notna().mean().mean(),'turn',f.rank(pct=True).diff().abs().mean().mean()); f.to_csv('scripts/miner_2_20260730_dxy_beta_signal.csv')
