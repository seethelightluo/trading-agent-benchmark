import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); ret=P.pct_change();
# defensive low volatility, lagged; use inverse vol with cross-sectional centering
sig=(-ret.rolling(40).std()).shift(1); fwd=P.shift(-10)/P-1
for h in [5,10,20,40]:
 fwd=P.shift(-h)/P-1;y=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:y.append(z.iloc[:,0].corr(z.iloc[:,1]))
 y=pd.Series(y).dropna();print(h,len(y),round(y.mean(),6),round(y.mean()/y.std(),6),round((y>0).mean(),4))
print('coverage',round(sig.notna().sum(axis=1).mean()/15,4));sig.to_csv('scripts/miner_2_20340721_lowvol_signal.csv',index_label='date')
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 y=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:y.append(z.iloc[:,0].corr(z.iloc[:,1]))
 y=pd.Series(y).dropna();print(a,b,len(y),round(y.mean(),6),round(y.mean()/y.std(),6))
